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

"""What two sides' models compare in: the one set of rules, said anywhere.

The analyses decide by these rules what of two sides pairs (token_analysis'
vocabulary_identities(), tensor_analysis.layout(), kv_analysis), and the
Session editor says by them, before a Session of the two models runs, what
will compare (verdict()); so the editor's hint can never say other than
what the analyses then do. Each rule reads only what a side's model is
(`Model`): its file's SHA-256, its runtime, its tokenizer's id -> piece table
(tap/model_facts) and its architecture, as the analyses have them.

- Tokens (logits, sampling): two sides of one model file share its
  vocabulary; two of different files share one when both tokenizers' tables
  are one (vocabularies()).
- Intermediate tensors (tensor_pairing()): two sides of one model file pair
  every tensor at its place (SAME_MODEL); two runtimes pair by the anchors
  of one architecture both models are of (BY_ANCHORS), and not at all when
  they are of different or unknown ones (NOT_PAIRED); two files of one
  runtime pair only by the pairs made by hand (BY_HAND).
- KV caches: two sides of one model file compare as stored (same_file());
  two of different files compare at each layer both store with one shape,
  which the capture shows. Before a run, two models of one skeleton are
  known to store one shape (same_skeleton()); of any other two it is not
  known.
"""

import dataclasses

LITERT_LM = 'LiteRT-LM'
# How two sides' intermediate tensors pair (tensor_pairing()).
SAME_MODEL = 'same_model'
BY_ANCHORS = 'by_anchors'
BY_HAND = 'by_hand'
NOT_PAIRED = 'not_paired'
# Why two sides' vocabularies are not one (Vocabularies.differ).
VOCABULARY_UNKNOWN = 'unknown'
VOCABULARY_DIFFERENT = 'different'
# The verdict's codes: what the editor's hint shows of two models.
SAME_FILE = 'same_file'
SAME_ARCHITECTURE = 'same_architecture'
DIFFERENT = 'different'
_SIDES = ('ref', 'target')


@dataclasses.dataclass(frozen=True)
class Model:
  """What a side's model is, as the rules compare it.

  Attributes:
      sha256: SHA-256 of its file (the source, not a copy that observes).
      runtime: The runtime it is of: LiteRT-LM or PyTorch.
      name: What the user knows it by.
      vocabulary: SHA-256 of its tokenizer's id -> piece table (model_facts'
        `table_sha256`), or None when it is not known.
      architecture: The id of its architecture (catalog), or None.
      architecture_name: That architecture's name; '' with none.
      skeleton: The skeleton of its architecture it is, or None.
      architecture_unknown: Why its architecture is not known yet, as a sentence
        (its graph is not converted, say); '' when it is known, or known to be
        none.
  """

  sha256: str
  runtime: str = LITERT_LM
  name: str = ''
  vocabulary: str | None = None
  architecture: str | None = None
  architecture_name: str = ''
  skeleton: str | None = None
  architecture_unknown: str = ''


@dataclasses.dataclass(frozen=True)
class Vocabularies:
  """Each side's vocabulary identity, and whether they are one.

  Attributes:
      identities: Per side, its identity: its model's SHA-256, or of two models
        whose tokenizers' tables are one, `tokenizer:<the table's SHA-256>`.
      differ: None when both are one; else VOCABULARY_UNKNOWN (a side's table is
        not known) or VOCABULARY_DIFFERENT.
      unknown: The sides whose tables are not known.
  """

  identities: dict[str, str]
  differ: str | None = None
  unknown: tuple[str, ...] = ()


def same_file(ref: Model, target: Model) -> bool:
  """Whether both sides are of one model file."""
  return ref.sha256 == target.sha256


def same_skeleton(ref: Model, target: Model) -> bool:
  """Whether both sides' models are known to be of one skeleton."""
  return (
      ref.architecture is not None
      and ref.skeleton is not None
      and (ref.architecture, ref.skeleton)
      == (target.architecture, target.skeleton)
  )


def vocabularies(ref: Model, target: Model) -> Vocabularies:
  """Each side's vocabulary identity: the one rule of which tokens pair."""
  models = {'ref': ref, 'target': target}
  files = {side: model.sha256 for side, model in models.items()}
  if same_file(ref, target):
    return Vocabularies(files)
  unknown = tuple(s for s in _SIDES if models[s].vocabulary is None)
  if unknown:
    return Vocabularies(files, VOCABULARY_UNKNOWN, unknown)
  identities = {
      side: f'tokenizer:{model.vocabulary}' for side, model in models.items()
  }
  if ref.vocabulary != target.vocabulary:
    return Vocabularies(identities, VOCABULARY_DIFFERENT)
  return Vocabularies(identities)


def tensor_pairing(ref: Model, target: Model) -> str:
  """How two sides' intermediate tensors pair: the one rule of it.

  Args:
      ref: The reference side's model.
      target: The target side's model.

  Returns:
      SAME_MODEL, BY_ANCHORS, BY_HAND or NOT_PAIRED, as the module says.
  """
  if ref.runtime != target.runtime:
    if ref.architecture is not None and (
        ref.architecture == target.architecture
    ):
      return BY_ANCHORS
    return NOT_PAIRED
  if not same_file(ref, target):
    return BY_HAND
  return SAME_MODEL


def verdict(ref: Model, target: Model) -> dict[str, str]:
  """What of two models compares, as the Session editor's hint says it.

  Args:
      ref: The reference side's model.
      target: The target side's model.

  Returns:
      `{code, text}`: SAME_FILE, SAME_ARCHITECTURE (two runtimes that pair
      by the anchors of one architecture) or DIFFERENT, and the hint.
  """
  if same_file(ref, target):
    return {
        'code': SAME_FILE,
        'text': 'Same model file: every analysis compares.',
    }
  tokens = vocabularies(ref, target)
  pairing = tensor_pairing(ref, target)
  compares = ['the generated text']
  notes = []
  if tokens.differ is None:
    compares += ['logits', 'sampling']
  elif tokens.differ == VOCABULARY_DIFFERENT:
    notes.append(
        'Their tokenizers map ids to different pieces, so logits and '
        'sampling do not compare.'
    )
  else:
    names = ' and '.join(
        ({'ref': ref, 'target': target}[side].name or side)
        for side in tokens.unknown
    )
    notes.append(
        f'The tokenizer of {names} cannot be read, so logits and '
        'sampling do not compare.'
    )
  if same_skeleton(ref, target):
    compares.append('KV caches')
  else:
    notes.append(
        'KV caches compare only at the layers both store in one shape.'
    )
  if pairing == BY_ANCHORS:
    compares.append("every layer's anchors")
  elif pairing == BY_HAND:
    notes.append(
        'Their intermediate tensors compare only as the pairs made by '
        'hand in Graph Diff.'
    )
  else:
    notes.append(_unpaired(ref, target))
  if pairing == BY_ANCHORS:
    label = ref.architecture_name or ref.architecture
    if same_skeleton(ref, target):
      label += f' · {ref.skeleton}'
    lead = f'Same architecture ({label})'
    if tokens.differ is None:
      lead += ' and the same tokenizer'
    code = SAME_ARCHITECTURE
  else:
    pending = ref.architecture_unknown or target.architecture_unknown
    if pairing == NOT_PAIRED and not pending:
      lead = 'Different architectures'
    elif pairing == BY_HAND:
      lead = 'Different model files of one runtime'
    else:
      lead = 'Different model files'
    if tokens.differ is None:
      lead += ' with the same tokenizer'
    code = DIFFERENT
  verb = 'compares' if len(compares) == 1 else 'compare'
  text = f'{lead}: {_listed(compares)} {verb}.'
  return {'code': code, 'text': ' '.join([text, *notes])}


def _unpaired(ref: Model, target: Model) -> str:
  """Why two runtimes' models' intermediate tensors do not pair."""
  unknown = [
      model.architecture_unknown
      for model in (ref, target)
      if model.architecture is None and model.architecture_unknown
  ]
  if unknown:
    return ' '.join(
        ['Whether their intermediate tensors pair is not known yet:'] + unknown
    )
  named = [
      f'{model.name} is '
      + (
          f'{model.architecture_name or model.architecture}'
          if model.architecture
          else 'of no architecture the server has'
      )
      for model in (ref, target)
  ]
  return (
      f'Their intermediate tensors do not pair: {named[0]}, {named[1]}; '
      'two runtimes pair by the anchors of one architecture.'
  )


def _listed(items: list[str]) -> str:
  """'a', 'a and b', 'a, b and c'."""
  if len(items) == 1:
    return items[0]
  return ', '.join(items[:-1]) + ' and ' + items[-1]
