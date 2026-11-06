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

"""A copy of a model that observes chosen tensors.

Observing a point (tap_scan) adds its tensor to the outputs of its signature,
and of the subgraph the signature runs, as `tap_sg<S>_op<O>_out<K>`; a tensor
the signature outputs already keeps its name. No operator is added or
changed: the runtime captures every output of a signature it runs, so the
runner writes an observed tensor as `<signature>_post_tap_sg<S>_op<O>_out<K>
_step_<N>.safetensors` at each step. The source model is never changed.

The copy is the source with its prefill and decode section rewritten
(litertlm_file.write_copy, which also checks the source against its SHA-256
as it reads it) and checked against the source before it is used
(check_copy): every byte outside the rewritten section, as
litertlm_file.check_copy() compares them; every weight buffer and large
custom option of the rewritten section byte for byte; and the whole object
tree of the rewritten model, every table, vector, union and value the TFLite
schema reads (operator codes; every buffer's fields; every subgraph's
tensors, with their type, shape, shape signature, buffer, quantization and
sparsity, its operators, with their builtin, second builtin and custom
options and intermediates, its inputs and outputs; the signature defs; the
metadata, metadata buffers, description and version) equal to the source's
but for the outputs added and the offsets of the payloads moved. The rewrite
serializes the section anew with the TFLite schema pinned in MODULE.bazel,
the one of the runtime's LiteRT (ci/check_schema_pins.py checks the pin):
what that schema lacks, the runtime does not read either, and a value of a
union member it lacks fails the check.

The copy is deterministic: the same source and points make the same bytes,
so its SHA-256, by which a runner keeps it, is the same each time. The only
metadata it changes is the system metadata's `uuid`, a name-based UUID of the
source's SHA-256 and the points.
"""

from collections.abc import Callable, Sequence
import dataclasses
import functools
import hashlib
import json
import pathlib
import re
import struct
from typing import Any, BinaryIO
import uuid

import flatbuffers
from model_debugger.server.tap import litertlm_file
from model_debugger.server.tap import litertlm_schema_generated
from model_debugger.server.tap import tap_scan
from model_debugger.server.tap import tflite_schema_generated as tflite
import numpy as np

# What a copy is: part of its uuid, and of what names it in a cache
# (identity()). Raised whenever this module writes other bytes for the same
# source and points; tap_model_test's digest of a copy fails until it is.
FORMAT_VERSION = 1
# The most points of one model a copy observes. A Session's default, every
# decoder layer's anchors in every signature that runs, is 189 of Gemma 4
# E2B (105 of decode alone); every operator of the layer ops menu (op_menu)
# ticked besides is 1,442 (760 of decode, 341 of each prefill signature),
# and a model of a fifth more layers (about 1,730) fits too. Each point
# is one more output of its signature, one more file at every step that
# signature runs, so a bound keeps a copy's outputs, and a step's files, to
# about two thousand.
MAX_POINTS = 2048
# What every observed output's name starts with; KV caches' names are the
# runtime's, and capture_index reads them.
OUTPUT_PREFIX = 'tap_'
# Out-of-line payloads (Buffer.offset) of the rewritten section start at
# multiples of this many bytes: this module's choice, a cache line, enough
# for every element type (the source's writer places them at 4 or 8).
_PAYLOAD_ALIGNMENT = 64
# Inline weights start at multiples of this many bytes: the schema's
# force_align of Buffer.data, which flatc's Python writer does not apply.
_DATA_ALIGNMENT = 16
_CHUNK = 8 * 1024 * 1024
# The namespace of the copies' uuids.
_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, 'model_debugger.server.tap')
# The capture key of an output a copy added (added_output_name).
_ADDED_KEY = re.compile(r'post_tap_sg(\d+)_op(\d+)_out(\d+)')


def added_output_name(point: tap_scan.Point) -> str:
  """The name of the output a copy adds to observe a point."""
  return f'{OUTPUT_PREFIX}sg{point.subgraph}_op{point.op}_out{point.output}'


def capture_key(point: tap_scan.Point) -> str:
  """The key of a point's tensor in a capture of a copy that observes it.

  `post_<output>`: the output its signature has already, or the one a copy
  adds (added_output_name).

  Args:
      point: The observed point.

  Returns:
      The point's tensor key in a capture.
  """
  return f'post_{point.output_name or added_output_name(point)}'


def added_point(key: str) -> tuple[int, int, int] | None:
  """The point a capture key names, if it is of an output a copy added.

  Args:
      key: A tensor's key in a capture file, such as `post_tap_sg0_op10_out0`.

  Returns:
      (subgraph, op, output) of the point; None for any other key.
  """
  match = _ADDED_KEY.fullmatch(key)
  return None if match is None else tuple(int(g) for g in match.groups())


@functools.cache
def identity() -> str:
  """What, besides the source and points, makes a copy's bytes.

  FORMAT_VERSION, the flatbuffers runtime's version, and the SHA-256 of
  each schema binding flatc wrote, so of the schemas pinned and of flatc:
  a copy kept by older code under another identity is not this code's.
  """
  digests = [
      hashlib.sha256(pathlib.Path(module.__file__).read_bytes()).hexdigest()
      for module in (tflite, litertlm_schema_generated)
  ]
  return ':'.join([str(FORMAT_VERSION), flatbuffers.__version__, *digests])


@dataclasses.dataclass(frozen=True)
class Observed:
  """A point a copy observes.

  Attributes:
      point: The point, as the source's scan lists it.
      output_name: The signature output that holds its tensor.
      added: Whether the copy added that output; False for an output the source
        has already.
  """

  point: tap_scan.Point
  output_name: str
  added: bool

  def capture_key(self) -> str:
    """The key of its tensor in a capture's files: `post_<output>`."""
    return f'post_{self.output_name}'


@dataclasses.dataclass(frozen=True)
class Copy:
  """A copy of a model that observes points.

  Attributes:
      sha256: SHA-256 of the copy.
      size: Its bytes.
      source_sha256: SHA-256 of the prefill and decode section it rewrote.
      observed: Its points, in the model's order.
      checked: How many of what check_copy() compared.
  """

  sha256: str
  size: int
  source_sha256: str
  observed: tuple[Observed, ...]
  checked: dict[str, int]


def copy_uuid(model_sha256: str, point_ids: Sequence[str]) -> str:
  """The `uuid` of the copy of a model that observes points.

  Args:
      model_sha256: SHA-256 of the source model file.
      point_ids: The points, in any order.

  Returns:
      A name-based UUID: the same for the same model and points.
  """
  name = json.dumps(
      [FORMAT_VERSION, model_sha256, sorted(point_ids)], separators=(',', ':')
  )
  return str(uuid.uuid5(_UUID_NAMESPACE, name))


def check_points(scan: tap_scan.Scan, point_ids: Sequence[str]) -> None:
  """Refuses points that a copy cannot observe.

  Args:
      scan: The model's scan.
      point_ids: The points chosen.

  Raises:
      ValueError: No points, too many, a point twice, one the model lacks,
        or one that cannot be observed.
  """
  if not 1 <= len(point_ids) <= MAX_POINTS:
    raise ValueError(f'Choose 1 to {MAX_POINTS} tensor outputs of each model.')
  if len(set(point_ids)) != len(point_ids):
    raise ValueError('A tensor output is chosen twice.')
  for point_id in point_ids:
    point = scan.point(point_id)
    if not point.selectable:
      raise ValueError(f'{point_id} cannot be observed: {point.reason}')


def write(
    source: pathlib.Path,
    scan: tap_scan.Scan,
    point_ids: Sequence[str],
    destination: pathlib.Path,
    model_sha256: str,
    progress: Callable[[str], None],
) -> Copy:
  """Writes and checks the copy of a model that observes points.

  Args:
      source: The model.
      scan: Its scan.
      point_ids: The points to observe, 1 to MAX_POINTS of them.
      destination: Where the copy goes; must not exist.
      model_sha256: SHA-256 of `source`, as the model store knows it; the source
        must still be that file.
      progress: Called with what is being done, as it starts.

  Returns:
      The copy.

  Raises:
      ValueError: The points are refused, the source is not the file
        `model_sha256` names, or the copy fails its check (it is then
        deleted).
  """
  check_points(scan, point_ids)
  # In the model's order, whatever the order chosen: the same points make
  # the same copy.
  points = sorted(
      (scan.point(point_id) for point_id in point_ids),
      key=lambda point: (point.subgraph, point.op, point.output),
  )
  try:
    with litertlm_file.mapped(source) as container:
      section = container.prefill_decode()
      progress(f'Adding {len(points)} outputs')
      # Not released: what refers to its bytes goes with `rewritten`,
      # and the map closes once the last view goes.
      rewritten, observed = _rewrite(container.bytes_of(section), points)
      progress('Writing the copy')
      with open(destination, 'xb') as out:
        litertlm_file.write_copy(
            container,
            section,
            rewritten.length,
            rewritten.write,
            copy_uuid(model_sha256, point_ids),
            model_sha256,
            out,
        )
      del rewritten
    progress('Checking the copy against the model')
    checked = check_copy(source, destination, observed)
    progress('Hashing the copy')
    sha256 = _hash(destination)
  except BaseException:
    destination.unlink(missing_ok=True)
    raise
  return Copy(
      sha256=sha256,
      size=destination.stat().st_size,
      source_sha256=scan.section_sha256,
      observed=tuple(observed),
      checked=checked,
  )


@dataclasses.dataclass
class _Rewritten:
  """A rewritten section: its FlatBuffer, then its relocated payloads.

  Attributes:
      header: The FlatBuffer.
      payloads: (where in the section, the source's bytes) of each out-of-line
        payload, in order.
      length: The section's bytes.
  """

  header: bytes
  payloads: list[tuple[int, 'memoryview[int]']]
  length: int

  def write(self, out: BinaryIO) -> None:
    """Writes the section from where `out` is."""
    base = out.tell()
    out.write(self.header)
    for target, payload in self.payloads:
      out.write(bytes(base + target - out.tell()))
      litertlm_file.copy_bytes(payload, out)


class AlignedBuffer(tflite.BufferT):
  """A Buffer whose inline data starts at a multiple of _DATA_ALIGNMENT."""

  @classmethod
  def of(cls, buffer: tflite.BufferT) -> 'AlignedBuffer':
    """The same buffer, packed aligned."""
    aligned = cls()
    aligned.data = buffer.data
    aligned.offset = buffer.offset
    aligned.size = buffer.size
    return aligned

  def Pack(self, builder: flatbuffers.Builder) -> int:
    """Packs it as BufferT does, its data aligned."""
    data = None
    if self.data is not None:
      payload = memoryview(self.data).cast('B')
      builder.StartVector(1, len(payload), _DATA_ALIGNMENT)
      builder.head -= len(payload)
      builder.Bytes[builder.head : builder.head + len(payload)] = payload
      data = builder.EndVector()
    tflite.BufferStart(builder)
    if data is not None:
      tflite.BufferAddData(builder, data)
    tflite.BufferAddOffset(builder, self.offset)
    tflite.BufferAddSize(builder, self.size)
    return tflite.BufferEnd(builder)


def _rewrite(
    data: 'memoryview[int]', points: list[tap_scan.Point]
) -> tuple[_Rewritten, list[Observed]]:
  """Rewrites a TFLite section to output the points' tensors.

  Args:
      data: The section.
      points: The points, each selectable.

  Returns:
      The rewritten section, which refers to `data`, and its points.

  Raises:
      ValueError: The model keeps weights in other files, a payload lies
        outside the section, or a point cannot be added.
  """
  if bytes(data[4:8]) != b'TFL3':
    raise ValueError('The prefill and decode section is not a TFLite model.')
  model = tflite.ModelT.InitFromObj(tflite.Model.GetRootAs(data, 0))
  if model.externalBuffers or model.externalBufferGroups:
    raise ValueError(
        'The model keeps weights in other files, which this server does '
        'not copy.'
    )
  observed = _promote(model, points)
  model.buffers = [AlignedBuffer.of(buffer) for buffer in model.buffers]
  payloads = []
  for buffer in model.buffers:
    if buffer.offset > 1:
      payloads.append((buffer, 'offset', buffer.offset, buffer.size))
  for graph in model.subgraphs:
    for op in graph.operators or []:
      if op.largeCustomOptionsOffset > 1:
        payloads.append((
            op,
            'largeCustomOptionsOffset',
            op.largeCustomOptionsOffset,
            op.largeCustomOptionsSize,
        ))
  for _, _, offset, size in payloads:
    # An empty payload is one: a zero-size weight still has its offset.
    if size < 0 or offset + size > len(data):
      raise ValueError('A weight of the model lies outside its section.')
  header = pack(model)
  if payloads:
    # Packed once to learn the FlatBuffer's size, which places the
    # payloads, and again with their new offsets, fixed-size fields.
    size = len(header)
    regions: dict[tuple[int, int], int] = {}
    cursor = _align(size, _PAYLOAD_ALIGNMENT)
    for item, field, offset, length in payloads:
      if (offset, length) not in regions:
        regions[(offset, length)] = cursor
        cursor = _align(cursor + length, _PAYLOAD_ALIGNMENT)
      setattr(item, field, regions[(offset, length)])
    header = pack(model)
    if len(header) != size:
      raise ValueError('The rewritten model changed size while placed.')
  else:
    regions = {}
  placed = [
      (target, data[offset : offset + length])
      for (offset, length), target in regions.items()
  ]
  end = max([len(header)] + [target + len(view) for target, view in placed])
  del model
  return _Rewritten(header, placed, end), observed


def _promote(
    model: tflite.ModelT, points: list[tap_scan.Point]
) -> list[Observed]:
  """Adds each point's tensor to its signature's and subgraph's outputs."""
  signatures = {
      _text(definition.signatureKey): definition
      for definition in model.signatureDefs or []
  }
  observed = []
  for point in points:
    definition = signatures[point.signature]
    graph = model.subgraphs[definition.subgraphIndex]
    tensor = int(graph.operators[point.op].outputs[point.output])
    if tensor != point.tensor:
      raise ValueError(f'{point.id} is not the tensor the scan found.')
    existing = [
        entry
        for entry in definition.outputs or []
        if entry.tensorIndex == tensor
    ]
    if len(existing) > 1:
      raise ValueError(
          f'{point.signature} outputs the tensor of {point.id} twice.'
      )
    if existing:
      name = _text(existing[0].name)
    else:
      name = added_output_name(point)
      if any(_text(entry.name) == name for entry in definition.outputs or []):
        raise ValueError(f'{point.signature} has an output {name}.')
      entry = tflite.TensorMapT()
      entry.name = name
      entry.tensorIndex = tensor
      definition.outputs = list(definition.outputs or []) + [entry]
    outputs = [int(index) for index in graph.outputs]
    if tensor not in outputs:
      graph.outputs = outputs + [tensor]
    observed.append(Observed(point, name, added=not existing))
  for name, definition in signatures.items():
    graph = model.subgraphs[definition.subgraphIndex]
    if {int(index) for index in graph.outputs} != {
        entry.tensorIndex for entry in definition.outputs or []
    }:
      raise ValueError(f'The outputs of {name} and of its subgraph differ.')
  return observed


def check_copy(
    source: pathlib.Path,
    copy: pathlib.Path,
    observed: Sequence[Observed],
) -> dict[str, int]:
  """Checks a copy against its source, as the module says.

  Args:
      source: The model.
      copy: Its copy.
      observed: The points the copy observes.

  Returns:
      How many sections, bytes outside the rewritten section, buffers,
      large custom options, observed points and values of the rewritten
      model's object tree were compared.

  Raises:
      ValueError: The copy differs from the source otherwise than by its
        observed outputs, the offsets moved and its uuid.
  """
  with (
      litertlm_file.mapped(source) as left,
      litertlm_file.mapped(copy) as right,
  ):
    replaced = left.prefill_decode()
    outside = litertlm_file.check_copy(left, right, replaced.index)
    # Not released: an error's traceback may hold views of them, and
    # the maps close once the last view goes.
    counts = _check_model(
        left.bytes_of(replaced),
        right.bytes_of(right.sections[replaced.index]),
        observed,
    )
  return {'sections': len(left.sections), 'outside_bytes': outside, **counts}


def _check_model(
    left: 'memoryview[int]',
    right: 'memoryview[int]',
    observed: Sequence[Observed],
) -> dict[str, int]:
  """Checks a rewritten TFLite section against its source.

  The payloads are compared byte for byte; then the object trees, with
  what the copy may change set aside: the outputs added, and the offsets
  of the payloads, which it moves.

  Args:
      left: The source TFLite section.
      right: The rewritten TFLite section.
      observed: The points the copy observes.

  Returns:
      The counts _check_trees() reports for what it compared.

  Raises:
      ValueError: The rewritten section differs from its source in more
        than what the copy may change.
  """
  raw_a = tflite.Model.GetRootAs(left, 0)
  raw_b = tflite.Model.GetRootAs(right, 0)
  a = tflite.ModelT.InitFromObj(raw_a)
  b = tflite.ModelT.InitFromObj(raw_b)
  try:
    return _check_trees(left, right, raw_a, raw_b, a, b, observed)
  except ValueError as error:
    message = str(error)
  # Raised anew: the trees hold views of both files, and the error's
  # traceback would keep them, and the files mapped, alive.
  del raw_a, raw_b, a, b
  raise ValueError(message)


def _check_trees(
    left: 'memoryview[int]',
    right: 'memoryview[int]',
    raw_a: tflite.Model,
    raw_b: tflite.Model,
    a: tflite.ModelT,
    b: tflite.ModelT,
    observed: Sequence[Observed],
) -> dict[str, int]:
  """_check_model() over both models, as tables and as object trees."""
  if len(a.buffers or []) != len(b.buffers or []):
    raise ValueError('The rewritten model has other buffers.')
  for i, (x, y) in enumerate(zip(a.buffers or [], b.buffers or [])):
    if (x.offset > 1) != (y.offset > 1) or not litertlm_file.same(
        _payload(left, raw_a.Buffers(i)), _payload(right, raw_b.Buffers(i))
    ):
      raise ValueError(
          f"Buffer {i} of the rewritten model differs from the source's."
      )
    # Compared: neither the bytes nor where they lie are compared again.
    x.data = y.data = None
    if x.offset > 1:
      x.offset = y.offset = 0
  if len(a.subgraphs or []) != len(b.subgraphs or []):
    raise ValueError('The rewritten model has other subgraphs.')
  added: dict[int, list[int]] = {}
  for item in observed:
    if item.added:
      added.setdefault(item.point.subgraph, []).append(item.point.tensor)
  custom = 0
  for i, (x, y) in enumerate(zip(a.subgraphs, b.subgraphs)):
    if len(x.operators or []) != len(y.operators or []):
      raise ValueError(
          f'Subgraph {i} of the rewritten model has other operators.'
      )
    for j, (p, q) in enumerate(zip(x.operators or [], y.operators or [])):
      if p.largeCustomOptionsOffset <= 1:
        continue
      custom += 1
      if q.largeCustomOptionsOffset <= 1 or not litertlm_file.same(
          _custom_options(left, raw_a.Subgraphs(i).Operators(j)),
          _custom_options(right, raw_b.Subgraphs(i).Operators(j)),
      ):
        raise ValueError(
            f'The custom options of operator {j} of subgraph {i} changed.'
        )
      p.largeCustomOptionsOffset = q.largeCustomOptionsOffset = 0
    before = _ints(x.outputs)
    expected = before + [t for t in added.get(i, []) if t not in before]
    if _ints(y.outputs) != expected:
      raise ValueError(f'Subgraph {i} has other outputs.')
    y.outputs = x.outputs
  outputs: dict[str, list[tuple[str, int]]] = {}
  for item in observed:
    if item.added:
      outputs.setdefault(item.point.signature, []).append(
          (item.output_name, item.point.tensor)
      )
  if len(a.signatureDefs or []) != len(b.signatureDefs or []):
    raise ValueError('The rewritten model has other signatures.')
  for x, y in zip(a.signatureDefs or [], b.signatureDefs or []):
    name = _text(x.signatureKey or b'')
    kept = len(x.outputs or [])
    extra = [
        (_text(entry.name), entry.tensorIndex)
        for entry in (y.outputs or [])[kept:]
    ]
    if extra != outputs.get(name, []):
      raise ValueError(f'Signature {name} has other outputs.')
    y.outputs = (y.outputs or [])[:kept] if x.outputs is not None else None
  values = _compare(a, b, 'the model')
  return {
      'buffers': len(a.buffers or []),
      'custom_options': custom,
      'points': len(observed),
      'values': values,
  }


def _compare(a: Any, b: Any, path: str) -> int:
  """Compares two object trees of the generated object API.

  Args:
      a: The source's.
      b: The copy's.
      path: Where they are, for the error.

  Returns:
      How many values were compared.

  Raises:
      ValueError: They differ, or a union holds a member the schema lacks.
  """
  if type(a) is not type(b):
    raise ValueError(f'The rewritten model changed {path}.')
  if a is None or isinstance(a, (bool, int, str, bytes)):
    if a != b:
      raise ValueError(f'The rewritten model changed {path}.')
    return 1
  if isinstance(a, float):
    # By their bits: NaN and -0.0 are values too.
    if struct.pack('<d', a) != struct.pack('<d', b):
      raise ValueError(f'The rewritten model changed {path}.')
    return 1
  if isinstance(a, np.ndarray):
    if a.dtype != b.dtype or a.shape != b.shape or (a.tobytes() != b.tobytes()):
      raise ValueError(f'The rewritten model changed {path}.')
    return 1
  if isinstance(a, list):
    if len(a) != len(b):
      raise ValueError(f'The rewritten model changed {path}.')
    return sum(
        _compare(x, y, f'{path}[{i}]') for i, (x, y) in enumerate(zip(a, b))
    )
  fields, others = vars(a), vars(b)
  if fields.keys() != others.keys():
    raise ValueError(f'The rewritten model changed {path}.')
  count = 0
  for name, value in fields.items():
    kind = fields.get(name + 'Type')
    if isinstance(kind, int) and kind and value is None:
      raise ValueError(f'{path}.{name} is of a kind the TFLite schema lacks.')
    count += _compare(value, others[name], f'{path}.{name}')
  return count


def _ints(vector: Any) -> list[int]:
  """A vector of ints of the object API: a list, an array, or None."""
  return [] if vector is None else [int(value) for value in vector]


def _custom_options(
    data: 'memoryview[int]', op: tflite.Operator
) -> 'memoryview[int]':
  """An operator's large custom options, out of line in its model."""
  return _span(data, op.LargeCustomOptionsOffset(), op.LargeCustomOptionsSize())


def _table(flatbuffer_object: Any) -> flatbuffers.table.Table:
  """Returns the raw table of a generated FlatBuffers object.

  The generated Python classes offer no public accessor for it.

  Args:
      flatbuffer_object: A generated FlatBuffers object.

  Returns:
      Its underlying table.
  """
  return flatbuffer_object._tab  # pylint: disable=protected-access


def _payload(
    data: 'memoryview[int]', buffer: tflite.Buffer
) -> 'memoryview[int]':
  """A buffer's bytes: out of line at its offset, or inline."""
  if buffer.Offset() > 1:
    return _span(data, buffer.Offset(), buffer.Size())
  field = _table(buffer).Offset(4)
  if not field:
    return data[0:0]
  start = _table(buffer).Vector(field)
  return data[start : start + _table(buffer).VectorLen(field)]


def _span(data: 'memoryview[int]', offset: int, size: int) -> 'memoryview[int]':
  if offset + size > len(data):
    raise ValueError('A weight lies outside the rewritten model.')
  return data[offset : offset + size]


def _text(value: bytes | str) -> str:
  """A string of the object API: bytes as read, str as this module sets."""
  return value.decode() if isinstance(value, bytes) else value


def pack(model: tflite.ModelT) -> bytes:
  """A TFLite model of the object API, serialized: its FlatBuffer."""
  builder = flatbuffers.Builder(1024)
  builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
  return bytes(builder.Output())


def _align(offset: int, alignment: int) -> int:
  return -(-offset // alignment) * alignment


def _hash(path: pathlib.Path) -> str:
  digest = hashlib.sha256()
  with open(path, 'rb') as file:
    while block := file.read(_CHUNK):
      digest.update(block)
  return digest.hexdigest()
