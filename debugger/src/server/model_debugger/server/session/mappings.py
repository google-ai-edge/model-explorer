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

"""Tensor pairs made by hand, kept per pair of models (T3).

The server pairs two sides' tensors by itself: one model's points alike, two
runtimes' by the anchors of their architecture and, for Gemma 4, by the
PyTorch module of each LiteRT-LM point's semantic operator
(graph/counterparts). In the Graph Diff's execution graph the user may also
pair a tensor of one side's graph with a tensor of the other's by hand:
an operator output of a signature's graph on each side. Such a pair belongs
to the two models, not to a Session: every Session of those two models, in
either order, has it.

The pairs of two models are one JSON file in the workspace's `mappings/`,
named by the two models' SHA-256s in sorted order, `<a>_<b>.json`, replaced
whole on each change, and deleted with either model (collect()); following
the user's own record of corresponding operators (pytorch-collector's
`rmsnorm_hf_vs_litert.json`, whose nodes name their counterparts on the
other side):

    {"format": 1, "ref": <sha256>, "target": <sha256>,
     "pairs": [{"id": ..., "label": ..., "created": ...,
                "ref": {"signature": ..., "node": ..., "output": ...,
                        "point": ...},
                "target": {...}}, ...]}

`ref` and `target` name which model each pair's `ref` and `target` tensor is
of: the order of the Session that saved the file's first pair. A Session
whose Reference loads the file's target model reads every pair the other way
round (pairs() turns them). A tensor is its signature, its node and output
in that signature's graph as the server serves it (model_graphs.graph()),
and the point that output carries there: a LiteRT-LM tap point, or a PyTorch
module's (what it returns or reads) or anchor's.

A pair is checked as it is saved (check()): each tensor is an output of its
graph that can be captured (its point observable); the two shapes hold the
same values per position, after a prefill's rows are dropped and without
axes of one, the same or the same elements with one of them flat; at least
one of the two is in a decoder layer, which Intermediate Tensors places it
by; and the pair is not one of the file's already. Two sides of one model
need none: each tensor is paired with itself (IDENTITY). What it pairs
automatically is refused where the store's caller knows it
(prepared_models).
"""

from collections.abc import Collection
import dataclasses
import pathlib
import re
import threading
from typing import Any
import uuid

from model_debugger.server.session import model_graphs as model_graphs_lib
from model_debugger.server.session import model_store
from model_debugger.server.session import workspace
from model_debugger.server.util import clock
from model_debugger.server.util import errors
from model_debugger.server.util import json_file

FORMAT = 1
# The most pairs two models keep: the editor adds one at a time.
MAX_PAIRS = 512
_LABEL_LIMIT = 200
_ID_LIMIT = 200
SIDES = ('ref', 'target')
SIDE_NAMES = {'ref': 'Reference', 'target': 'Target'}
IDENTITY = (
    'Reference and Target load the same model: each tensor is paired with '
    'itself, so no pair is made by hand.'
)
_SHAPE = re.compile(r'^[^\[]*\[([^\]]*)\]$')


@dataclasses.dataclass(frozen=True)
class Tensor:
  """One side's tensor of a pair: an operator output of a graph.

  Attributes:
      signature: The signature whose graph it is of.
      node: Its operator's node there.
      output: Its output's id there.
      point: The point that output carries: what is captured of it.
  """

  signature: str
  node: str
  output: str
  point: str

  def to_json(self) -> dict[str, Any]:
    """The tensor as a pair's file keeps it."""
    return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Pair:
  """A pair made by hand, of a Reference and a Target tensor.

  Attributes:
      id: Its id, the store's own.
      ref: The Reference's tensor.
      target: The Target's tensor.
      label: What the user named it; '' if nothing.
      created: When it was saved (clock.now()).
  """

  id: str
  ref: Tensor
  target: Tensor
  label: str
  created: str

  def flipped(self) -> 'Pair':
    """The pair the other way round.

    As a Session of the two models swapped reads it.

    Returns:
      A copy of the pair with ref and target tensors swapped.
    """
    return dataclasses.replace(self, ref=self.target, target=self.ref)

  def to_json(self) -> dict[str, Any]:
    """The pair as its file keeps it."""
    return {
        'id': self.id,
        'label': self.label,
        'created': self.created,
        'ref': self.ref.to_json(),
        'target': self.target.to_json(),
    }


def _tensor(value: dict[str, Any]) -> Tensor:
  return Tensor(
      signature=str(value['signature']),
      node=str(value['node']),
      output=str(value['output']),
      point=str(value['point']),
  )


def _pair(value: dict[str, Any]) -> Pair:
  return Pair(
      id=str(value['id']),
      ref=_tensor(value['ref']),
      target=_tensor(value['target']),
      label=str(value.get('label', '')),
      created=str(value.get('created', '')),
  )


def file_name(ref_sha256: str, target_sha256: str) -> str:
  """The name of the file of two models' pairs, in `mappings/`."""
  first, second = sorted((ref_sha256, target_sha256))
  return f'{first}_{second}.json'


class MappingStore:
  """The pairs made by hand of every pair of models of a workspace.

  `revision` counts the changes since the server started, so that what is
  worked out of the pairs (a Session's view) is made again after one.
  """

  def __init__(self, workspace_: workspace.Workspace) -> None:
    """Keeps the pairs in `workspace_.mappings`."""
    self._directory = workspace_.mappings
    self._lock = threading.Lock()
    self._revision = 0

  @property
  def revision(self) -> int:
    """How many changes were saved since the store was opened."""
    with self._lock:
      return self._revision

  def pairs(self, ref_sha256: str, target_sha256: str) -> list[Pair]:
    """The pairs of two models, as a Session of them in that order has them.

    [] for one model.

    Args:
      ref_sha256: The Reference model SHA256.
      target_sha256: The Target model SHA256.

    Returns:
      The list of pairs for the two models.

    Raises:
      ValueError: The file is of another format.
    """
    if ref_sha256 == target_sha256:
      return []
    with self._lock:
      return self._oriented(ref_sha256, target_sha256)

  def add(
      self,
      ref_sha256: str,
      target_sha256: str,
      ref: Tensor,
      target: Tensor,
      label: str = '',
  ) -> Pair:
    """Saves a pair of two models' tensors.

    Args:
        ref_sha256: The Reference's model.
        target_sha256: The Target's model.
        ref: The Reference's tensor, checked (check()).
        target: The Target's tensor, checked.
        label: What the user names it; '' for nothing.

    Returns:
        The pair saved, as pairs() reads it in that order.

    Raises:
        ValueError: One model, a label too long, or MAX_PAIRS saved.
        Conflict: The two models have that pair already.
    """
    if ref_sha256 == target_sha256:
      raise ValueError(IDENTITY)
    label = _label(label)
    with self._lock:
      stored = self._oriented(ref_sha256, target_sha256)
      for pair in stored:
        if pair.ref.point == ref.point and (pair.target.point == target.point):
          raise errors.Conflict(
              f'{ref.point} and {target.point} are paired already'
              f'{f" ({pair.label})" if pair.label else ""}.'
          )
      if len(stored) >= MAX_PAIRS:
        raise ValueError(
            f'Two models keep at most {MAX_PAIRS} pairs; delete one first.'
        )
      pair = Pair(
          id=uuid.uuid4().hex[:12],
          ref=ref,
          target=target,
          label=label,
          created=clock.now(),
      )
      self._write(ref_sha256, target_sha256, [*stored, pair])
      return pair

  def delete(self, ref_sha256: str, target_sha256: str, pair_id: str) -> Pair:
    """Deletes a pair of two models.

    Args:
      ref_sha256: The Reference's model.
      target_sha256: The Target's model.
      pair_id: The ID of the pair to delete.

    Returns:
      The pair deleted, as pairs() read it.

    Raises:
      NotFound: The two models have no such pair.
    """
    with self._lock:
      stored = (
          self._oriented(ref_sha256, target_sha256)
          if ref_sha256 != target_sha256
          else []
      )
      kept = [pair for pair in stored if pair.id != pair_id]
      if len(kept) == len(stored):
        raise errors.NotFound(
            f'These two models have no pair {str(pair_id)[:_ID_LIMIT]}.'
        )
      self._write(ref_sha256, target_sha256, kept)
      return next(pair for pair in stored if pair.id == pair_id)

  def collect(self, kept: Collection[str]) -> list[str]:
    """Deletes the pairs of every two models one of which is gone.

    As a model is deleted (model_store.ModelStore.collect()), so are the
    pairs made by hand of it with any other model.

    Args:
        kept: The SHA-256s of the models the workspace keeps.

    Returns:
        The names of the files deleted.
    """
    removed = []
    with self._lock:
      if not self._directory.is_dir():
        return removed
      for path in sorted(self._directory.glob('*_*.json')):
        shas = path.stem.split('_')
        if len(shas) == 2 and all(sha in kept for sha in shas):
          continue
        path.unlink(missing_ok=True)
        removed.append(path.name)
      if removed:
        self._revision += 1
    return removed

  def _path(self, ref_sha256: str, target_sha256: str) -> pathlib.Path:
    return self._directory / file_name(ref_sha256, target_sha256)

  def _oriented(self, ref_sha256: str, target_sha256: str) -> list[Pair]:
    """The stored pairs, Reference first; called with the lock held."""
    path = self._path(ref_sha256, target_sha256)
    if not path.is_file():
      return []
    stored = json_file.read_json(path)
    if not isinstance(stored, dict) or stored.get('format') != FORMAT:
      raise ValueError(
          f'The tensor pairs in {path.name} were saved in another format.'
      )
    found = [_pair(value) for value in stored.get('pairs', [])]
    if stored.get('ref') == ref_sha256:
      return found
    return [pair.flipped() for pair in found]

  def _write(
      self, ref_sha256: str, target_sha256: str, pairs: list[Pair]
  ) -> None:
    """Writes the pairs, Reference first; called with the lock held.

    A file keeps the order it was first written in.

    Args:
      ref_sha256: The Reference's model.
      target_sha256: The Target's model.
      pairs: The list of pairs to write.
    """
    path = self._path(ref_sha256, target_sha256)
    order = ref_sha256, target_sha256
    if path.is_file():
      stored = json_file.read_json(path)
      if stored.get('ref') == target_sha256:
        order = target_sha256, ref_sha256
        pairs = [pair.flipped() for pair in pairs]
    if not pairs:
      path.unlink(missing_ok=True)
    else:
      json_file.write_json(
          path,
          {
              'format': FORMAT,
              'ref': order[0],
              'target': order[1],
              'pairs': [pair.to_json() for pair in pairs],
          },
      )
    self._revision += 1


def _label(value: Any) -> str:
  """Checks a pair's label: text of at most _LABEL_LIMIT characters."""
  if value is None:
    return ''
  if (
      not isinstance(value, str)
      or len(value) > _LABEL_LIMIT
      or any(ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF for char in value)
  ):
    raise ValueError(
        'A pair label must be text with no control characters, of at most '
        f'{_LABEL_LIMIT} characters.'
    )
  return value.strip()


def shape_of(text: str) -> tuple[int, ...] | None:
  """A graph output's shape from its `tensor_shape`, if it names one.

  Such as `float32[1, 1, 2048]`; None if it names none.

  Args:
    text: The tensor shape text to parse.

  Returns:
    The shape tuple if parsed, or None.
  """
  match = _SHAPE.match(text.strip())
  if match is None:
    return None
  try:
    return tuple(
        int(size) for size in match.group(1).split(',') if size.strip()
    )
  except ValueError:
    return None


def compatible(first: tuple[int, ...], second: tuple[int, ...]) -> bool:
  """Whether two tensors' shapes hold the same values per position.

  Without axes of one: the same, or the same elements with one of them
  flat. Where both are [1, rows, ...] with more than one row (a prefill's
  rows, its tokens, which Intermediate Tensors compares one at a time),
  the rows are dropped first.

  Args:
    first: The first shape tuple.
    second: The second shape tuple.

  Returns:
    True if the shapes are compatible, False otherwise.
  """

  def kept(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(size for size in shape if size != 1)

  def same(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    a, b = kept(a), kept(b)
    if a == b:
      return True
    size = 1
    for extent in a:
      size *= extent
    other = 1
    for extent in b:
      other *= extent
    return size == other and min(len(a), len(b)) <= 1

  if same(first, second):
    return True
  rows = [
      len(shape) >= 2 and shape[0] == 1 and shape[1] > 1
      for shape in (first, second)
  ]
  return all(rows) and same(first[2:], second[2:])


def requested(value: Any, side: str) -> tuple[str, str, str]:
  """A tensor of a pair as the editor sends it: (signature, node, output).

  Args:
    value: The dictionary containing signature, node, and output keys.
    side: The side name ('ref' or 'target').

  Returns:
    A tuple of (signature, node, output).

  Raises:
    ValueError: It is not `{signature, node, output}` of texts.
  """
  if not isinstance(value, dict) or not all(
      isinstance(value.get(key), str)
      and value.get(key)
      and len(value[key]) <= _ID_LIMIT
      for key in ('signature', 'node', 'output')
  ):
    raise ValueError(
        f'Name the {SIDE_NAMES[side]} tensor as {{signature, node, '
        'output}: an operator output of its graph.'
    )
  return value['signature'], value['node'], value['output']


@dataclasses.dataclass(frozen=True)
class Checked:
  """A tensor of a pair, as its graph serves it.

  Attributes:
      tensor: What a pair keeps of it.
      layer: Its layer, if its graph places it in one.
      shape: Its shape in the graph.
      name: Its tensor's name there; '' if it has none (a PyTorch module's,
        named by its point).
  """

  tensor: Tensor
  layer: int | None
  shape: tuple[int, ...]
  name: str = ''


_MODULE_POINT = re.compile(r'^[^:]+:(.+):(in|out)$')


def default_label(checked: dict[str, Checked]) -> str:
  """What a pair saved without a label is called: its two tensors.

  A tensor's name by its last two scopes (`attn_vec_einsum/reshape`); a
  PyTorch module's path inside its layer and which of it
  (`self_attn.o_proj input`); else its point.

  Args:
    checked: Dict mapping side name ('ref', 'target') to Checked tensor info.

  Returns:
    The formatted default label string.
  """

  def short(found: Checked) -> str:
    if found.name:
      return '/'.join(found.name.split(';')[0].split('/')[-2:])
    match = _MODULE_POINT.match(found.tensor.point)
    if match is None:
      return found.tensor.point
    inside = re.sub(r'^.*?\blayers\.\d+\.?', '', match.group(1)) or 'layer'
    return f'{inside} {"input" if match.group(2) == "in" else "output"}'

  return f'{short(checked["ref"])} ↔ {short(checked["target"])}'


def check(
    graphs: model_graphs_lib.ModelGraphs,
    models: dict[str, model_store.Model],
    tensors: dict[str, tuple[str, str, str]],
) -> dict[str, Checked]:
  """Checks a pair's two tensors against their models' graphs.

  Args:
      graphs: The models' graphs.
      models: Per side, its model.
      tensors: Per side, its tensor as requested().

  Returns:
      Per side, the tensor as its graph serves it.

  Raises:
      ValueError: One model, a tensor its graph does not have or that
        cannot be captured, shapes that hold different values, or two
        tensors in no decoder layer; each in the user's words.
      Conflict: A model's graph is not converted.
  """
  if models['ref'].sha256 == models['target'].sha256:
    raise ValueError(IDENTITY)
  found = {}
  for side in SIDES:
    signature, node, output = tensors[side]
    model = models[side]
    try:
      facts = graphs.tensor(model, signature, node, output)
    except errors.NotFound as error:
      raise ValueError(f'{SIDE_NAMES[side]} ({model.name}): {error}') from None
    where = f'{SIDE_NAMES[side]} {signature} node {node}:{output}'
    if not facts.get('point_id') or facts.get('observable') != 'true':
      reason = facts.get('reason') or 'it is no point of the model'
      raise ValueError(f'{where} cannot be captured: {reason}')
    shape = shape_of(facts.get('tensor_shape', ''))
    if shape is None:
      raise ValueError(f'{where} has no shape in its graph.')
    layer = facts.get('layer')
    found[side] = Checked(
        Tensor(signature, node, output, facts['point_id']),
        int(layer) if layer and layer.isdigit() else None,
        shape,
        facts.get('tensor_name', ''),
    )
  if not compatible(found['ref'].shape, found['target'].shape):
    raise ValueError(
        f'Reference {found["ref"].tensor.point} is '
        f'{list(found["ref"].shape)} and Target '
        f'{found["target"].tensor.point} {list(found["target"].shape)}: '
        'they do not hold the same values per position, even without '
        'axes of one or flattened.'
    )
  if found['ref'].layer is None and found['target'].layer is None:
    raise ValueError(
        'Neither tensor is in a decoder layer: Intermediate Tensors '
        'places each pair in its layer.'
    )
  return found
