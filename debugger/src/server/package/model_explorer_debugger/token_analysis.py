# Copyright 2026 The AI Edge Model Explorer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Per-token evidence, never batch aggregates or a sampled top-k approximation.

Pairs name original generation steps on each side. A single sampled token must
be linked to a completed forward before its final output position can be used.
Full logits use the captured vocabulary digest. Activation metrics require an
explicit last_hidden_state boundary; logits are not relabelled as activations.
"""

from collections import OrderedDict
import hashlib
import math
from pathlib import Path
import threading
import numpy as np

KEYS = (
    'kl',
    'js',
    'relative_l2',
    'cosine_distance',
    'norm_ratio',
    'max_abs_error',
)


def missing(reason):
  return {'value': None, 'reason': reason}


def vector_metrics(ref, target):
  if ref.shape != target.shape:
    return {key: missing('Activation shapes differ') for key in KEYS[2:]}
  a, b = ref.astype(np.float64), target.astype(np.float64)
  scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
  if not np.isfinite(scale):
    return {
        key: missing('Activation contains non-finite values')
        for key in KEYS[2:]
    }
  aa, bb = (a / scale, b / scale) if scale else (a, b)
  na, nb = float(np.linalg.norm(aa)), float(np.linalg.norm(bb))
  values = {
      'relative_l2': 100 * float(np.linalg.norm(bb - aa)) / na if na else None,
      'norm_ratio': nb / na if na else None,
      'cosine_distance': (
          float(np.clip(1 - np.dot(aa, bb) / (na * nb), 0, 2))
          if na and nb
          else None
      ),
      'max_abs_error': float(np.max(np.abs(bb - aa))) * scale,
  }
  return {
      key: (
          {'value': value, 'reason': None}
          if value is not None and math.isfinite(value)
          else missing('Zero reference norm or non-finite result')
      )
      for key, value in values.items()
  }


class TokenAnalysis:

  def __init__(self, store):
    self.store = store
    self.lock = threading.Lock()
    data = store._telemetry
    self.forwards = {
        (r['turn'], r['run'], r['forward_id']): r for r in data['forwards']
    }
    self.events = {}
    self.tokens = {}
    self.labels = {}
    for row in store.session.get('conversation', []):
      for token in row.get('tokens', []):
        self.tokens[row['turn'], row['run'], token['step']] = token
        if type(token.get('id')) is int:
          self.labels[row['turn'], row['run'], token['id']] = token.get('text')
    for row in data['token_records']:
      if row.get('candidate') != 0 or type(row.get('output_index')) is not int:
        continue
      for offset, token_id in enumerate(row['token_ids']):
        self.events.setdefault(
            (row['turn'], row['run'], row['output_index'] + offset), []
        ).append((row, token_id))
    self.vectors = OrderedDict()
    self.results = OrderedDict()
    self._tokenizer = None
    self._tokenizer_loaded = False

  def _load_tokenizer(self):
    """Loads the exact SentencePiece model saved with the capture.

    Returns None when it is absent or unreadable.
    """
    try:
      import sentencepiece as spm
    except ImportError:
      return None
    root = getattr(self.store, 'root', None)
    if root is None:
      return None
    root = Path(root).resolve()
    for run in self.store.session.get('runs', []):
      info = run.get('tokenizer') or {}
      relative = info.get('path')
      if (
          info.get('kind') != 'sentencepiece'
          or not isinstance(relative, str)
          or Path(relative).is_absolute()
      ):
        continue
      path = (root / relative).resolve()
      if not path.is_relative_to(root) or not path.is_file():
        continue
      blob = path.read_bytes()
      if (
          info.get('sha256')
          and hashlib.sha256(blob).hexdigest() != info['sha256']
      ):
        continue
      try:
        return spm.SentencePieceProcessor(model_proto=blob)
      except Exception:
        continue
    return None

  def label(self, turn, identity):
    """Returns text for a vocabulary ID.

    Prefers the generated token's own text, else the saved tokenizer's.
    """
    text = self.labels.get(
        (turn, 'ref', identity), self.labels.get((turn, 'target', identity))
    )
    if text is not None:
      return text
    if not self._tokenizer_loaded:
      self._tokenizer, self._tokenizer_loaded = self._load_tokenizer(), True
    tokenizer = self._tokenizer
    if (
        tokenizer is None
        or type(identity) is not int
        or not 0 <= identity < tokenizer.GetPieceSize()
    ):
      return None
    # Piece form, with SentencePiece's word-boundary marker shown as the space
    # it stands for; this matches the text the runtime releases (' body', ',')
    # regardless of decoder options.
    piece = tokenizer.IdToPiece(identity)
    return (
        piece
        if tokenizer.IsControl(identity) or tokenizer.IsByte(identity)
        else piece.replace('\u2581', ' ')
    )

  def vector(self, forward, name):
    # The native runtime labels the same final hidden state `activations`.
    names = (
        (name, 'activations')
        if name == 'last_hidden_state' and forward.get('runtime') == 'LiteRT-LM'
        else (name,)
    )
    matches = [
        r
        for r in forward.get('outputs', [])
        if r.get('output_path') in [['output', n] for n in names]
        and r.get('resource_id')
    ]
    if len(matches) != 1:
      raise ValueError(
          (
              'Full-vocabulary logits'
              if name == 'logits'
              else 'Last hidden-state activation'
          )
          + ' not captured'
      )
    identity = matches[0]['resource_id']
    if identity in self.vectors:
      self.vectors.move_to_end(identity)
      return self.vectors[identity]
    tensor = np.asarray(self.store.load_resource(identity))
    # Supported causal-LM boundary layouts. Batch/sequence axes are never
    # flattened into vocabulary.
    if tensor.ndim == 3 and tensor.shape[0] == 1 and tensor.shape[1] > 0:
      vector = tensor[0, -1]
    elif tensor.ndim == 2 and tensor.shape[0] == 1:
      vector = tensor[0]
    elif tensor.ndim == 1:
      vector = tensor
    else:
      raise ValueError('Token output layout is not a single causal-LM sequence')
    if not vector.size or (
        vector.dtype.kind not in 'fiu' and str(vector.dtype) != 'bfloat16'
    ):
      raise ValueError('Token output is not a numeric vector')
    vector = vector.astype(np.float64)
    self.vectors[identity] = vector
    while len(self.vectors) > 8:
      self.vectors.popitem(last=False)
    return vector

  def side(self, turn, run, step):
    if step is None:
      raise ValueError('Token not captured on this side')
    token = self.tokens.get((turn, run, step))
    events = self.events.get((turn, run, step), [])
    if not token or len(events) != 1 or events[0][1] != token.get('id'):
      raise ValueError('Unique sampled-token source not captured')
    event, _ = events[0]
    # A multi-output chunk does not prove which logit position sampled each.
    if len(event['token_ids']) != 1 or (
        token.get('source_forward_id') is not None
        and token['source_forward_id'] != event['forward_id']
    ):
      raise ValueError('Sampled-token output position is ambiguous')
    forward = self.forwards.get((turn, run, event['forward_id']))
    if not forward or forward.get('status') != 'completed':
      raise ValueError('Completed source forward not captured')
    return token, forward

  def distribution(self, turn, run, token, forward):
    logits = self.vector(forward, 'logits')
    if (
        np.any(np.isnan(logits))
        or np.any(np.isposinf(logits))
        or not np.any(np.isfinite(logits))
    ):
      raise ValueError('Logits contain invalid values')
    shifted = logits - np.max(logits)
    logp = shifted - np.log(np.exp(shifted).sum())
    p = np.exp(logp)
    order = np.lexsort((np.arange(p.size), -p))
    ranks = np.empty(p.size, dtype=np.int64)
    ranks[order] = np.arange(1, p.size + 1)
    selected = token.get('id')
    selected = (
        selected if type(selected) is int and 0 <= selected < p.size else None
    )
    mask = p > 0
    # The margin between the two most likely tokens says how close this step was
    # to choosing differently; it is a property of one side's own distribution,
    # valid in any context.
    summary = {
        'entropy': float(-np.sum(p[mask] * logp[mask]) / np.log(2)),
        'selected_id': selected,
        'selected_rank': int(ranks[selected]) if selected is not None else None,
        'selected_probability': (
            float(p[selected]) if selected is not None else None
        ),
        'margin': float(p[order[0]] - p[order[1]]) if p.size > 1 else None,
        'size': int(p.size),
        'source_forward_id': forward['forward_id'],
        'reason': (
            None
            if selected is not None
            else 'Selected token ID is outside captured vocabulary'
        ),
    }
    vocab = forward.get('vocab_identity') or forward.get('input_proof', {}).get(
        'tokenizer', {}
    ).get('vocab_sha256')
    return dict(
        summary=summary,
        p=p,
        logp=logp,
        logits=logits,
        order=order,
        ranks=ranks,
        vocab=vocab,
    )

  def pair(self, turn, ref_step, target_step):
    key = turn, ref_step, target_step
    if key in self.results:
      self.results.move_to_end(key)
      return self.results[key]
    metrics = {
        key: missing('Paired token evidence not captured') for key in KEYS
    }
    distributions, forwards, reasons = {}, {}, {}
    for run, step in (('ref', ref_step), ('target', target_step)):
      try:
        token, forward = self.side(turn, run, step)
        forwards[run] = forward
        distributions[run] = self.distribution(turn, run, token, forward)
      except (ValueError, KeyError, OSError, TypeError) as error:
        reasons[run] = str(error)
    ref, target = distributions.get('ref'), distributions.get('target')
    vocabulary = bool(
        ref
        and target
        and ref['vocab']
        and ref['vocab'] == target['vocab']
        and ref['p'].shape == target['p'].shape
    )
    compatible = vocabulary
    reason = (
        'Matching captured vocabulary IDs and sizes are required'
        if ref and target
        else '; '.join(f'{run}: {value}' for run, value in reasons.items())
    )
    native_context = True
    if len(forwards) == 2 and any(
        row.get('runtime') == 'LiteRT-LM' for row in forwards.values()
    ):
      a, b = forwards['ref'], forwards['target']
      native_context = bool(a.get('sample') and a['sample'] == b.get('sample'))
      if not native_context:
        compatible = False
        reason = (
            'Matching captured logical token contexts are required; generated'
            ' histories have diverged or are unproven'
        )
    if compatible:
      p, q = ref['p'], target['p']
      positive = p > 0
      kl = float(
          np.sum(
              p[positive] * (ref['logp'][positive] - target['logp'][positive])
          )
      )
      m = (p + q) / 2
      js = sum(
          float(np.sum(a[a > 0] * (np.log2(a[a > 0]) - np.log2(m[a > 0]))))
          * 0.5
          for a in (p, q)
      )
      metrics['kl'] = (
          {'value': max(0, kl), 'reason': None}
          if math.isfinite(kl)
          else missing(
              'Divergence is infinite: target assigns zero probability'
          )
      )
      metrics['js'] = {'value': float(np.clip(js, 0, 1)), 'reason': None}
    else:
      metrics.update({
          key: missing(reason or 'Full-vocabulary logits not captured')
          for key in ('kl', 'js')
      })
    try:
      if len(forwards) != 2:
        raise ValueError('Paired source forwards not captured')
      a, b = forwards['ref'], forwards['target']
      if not native_context:
        raise ValueError(reason)
      if not a.get('model_sha256') or a['model_sha256'] != b.get(
          'model_sha256'
      ):
        raise ValueError(
            'Matching activation coordinate spaces are not established'
        )
      metrics.update(
          vector_metrics(
              self.vector(a, 'last_hidden_state'),
              self.vector(b, 'last_hidden_state'),
          )
      )
    except (ValueError, KeyError, OSError, TypeError) as error:
      metrics.update({key: missing(str(error)) for key in KEYS[2:]})
    # Candidate rows are keyed by vocabulary ID, so they need one vocabulary,
    # not one context: after the generations diverge each side still shows its
    # own candidates, without a delta.
    rows = []
    if not (ref and target and not vocabulary):
      sides = [s for s in (ref, target) if s]
      pinned = list(
          dict.fromkeys(
              s['summary']['selected_id']
              for s in sides
              if s['summary']['selected_id'] is not None
          )
      )
      pool = set(int(i) for s in sides for i in s['order'][:64]) - set(pinned)
      ids = (
          pinned
          + sorted(
              pool, key=lambda i: (-max(float(s['p'][i]) for s in sides), i)
          )
      )[:64]
      for identity in ids:
        row = {
            'id': identity,
            'label': self.label(turn, identity),
            'delta': None,
        }
        for run, value in (('ref', ref), ('target', target)):
          logit = float(value['logits'][identity]) if value else None
          row[run] = (
              None
              if not value
              else {
                  'probability': float(value['p'][identity]),
                  'rank': int(value['ranks'][identity]),
                  'logit': logit if math.isfinite(logit) else '-Infinity',
                  'selected': value['summary']['selected_id'] == identity,
              }
          )
        if compatible:
          row['delta'] = float(
              (target['p'][identity] - ref['p'][identity]) * 100
          )
        rows.append(row)
    basis = {
        run: forward.get('basis', 'recorded')
        for run, forward in forwards.items()
    }
    result = {
        'ref_step': ref_step,
        'target_step': target_step,
        'metrics': metrics,
        'basis': (
            basis.get('ref')
            if len(set(basis.values())) == 1
            else (basis or None)
        ),
        'distribution': {
            'compatible': compatible,
            'reason': None if compatible else reason,
            'context': (
                None
                if not (ref and target)
                else 'same'
                if native_context
                else 'different'
            ),
            'ref': ref['summary'] if ref else {'reason': reasons.get('ref')},
            'target': (
                target['summary']
                if target
                else {'reason': reasons.get('target')}
            ),
            'rows': rows,
        },
    }
    self.results[key] = result
    while len(self.results) > 512:
      self.results.popitem(last=False)
    return result


def analyze_tokens(store, payload):
  turn, pairs = payload.get('turn'), payload.get('pairs')
  if (
      type(turn) is not int
      or turn < 1
      or not isinstance(pairs, list)
      or len(pairs) > 128
  ):
    raise ValueError('Expected one turn and at most 128 token pairs')
  for pair in pairs:
    if (
        not isinstance(pair, dict)
        or set(pair) != {'ref', 'target'}
        or any(
            s is not None and (type(s) is not int or s < 0)
            for s in pair.values()
        )
    ):
      raise ValueError('Expected original ref and target generation steps')
  # Store lifetime is one immutable capture. No cross-session cache.
  if not hasattr(store, '_token_analysis'):
    store._token_analysis = TokenAnalysis(store)
  analyzer = store._token_analysis
  with analyzer.lock:
    return {
        'turn': turn,
        'pairs': [analyzer.pair(turn, p['ref'], p['target']) for p in pairs],
    }
