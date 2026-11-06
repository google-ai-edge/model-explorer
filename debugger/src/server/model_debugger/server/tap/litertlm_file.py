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

"""A .litertlm file: its header and sections, and a copy with one replaced.

The layout, as LiteRT-LM reads it (schema/core/litertlm_read.cc): the magic
`LITERTLM`, three little-endian uint32 versions, 4 bytes of padding, the
uint64 offset where the header ends, and from byte 32 the header itself, a
LiteRTLMMetaData FlatBuffer: the system metadata (key-value pairs) and, per
section, its metadata, data type and [begin, end) byte range. Sections start
at multiples of 16 KiB.

A copy keeps every byte of the source but one section's, the padding after
it (zeros, in the source as in the copy), the offsets of that section and of
those after it, and the system metadata's `uuid`, which names the file: the
copy's header is the source's own, patched in place, not serialized anew, so
nothing a newer writer added to it is lost; what follows the section, every
later section and every byte between them and after them, moves by a whole
number of blocks.
"""

from collections.abc import Callable, Iterator
import contextlib
import dataclasses
import hashlib
import mmap
import pathlib
import struct
from typing import Any, BinaryIO

import flatbuffers
from model_debugger.server.tap import litertlm_schema_generated as schema

MAGIC = b'LITERTLM'
# Where the header's FlatBuffer starts, and where its end is written.
HEADER_BEGIN = 32
_HEADER_END_AT = 24
BLOCK_BYTES = 16 * 1024
# The model types of the section that holds the text model's prefill and
# decode signatures: the TF-free name, and the one older files carry.
PREFILL_DECODE = ('prefill_decode', 'tf_lite_prefill_decode')
# Bytes copied at a time.
_CHUNK = 8 * 1024 * 1024
# The vtable slots of SectionObject's begin_offset and end_offset, and of
# StringValue's value (the schema's field order).
_BEGIN_SLOT = 6
_END_SLOT = 8
_STRING_SLOT = 4
_UINT64 = struct.Struct('<Q')


@dataclasses.dataclass(frozen=True)
class Section:
  """One section of a .litertlm file.

  Attributes:
      index: Its place in the header.
      data_type: Its schema.AnySectionDataType.
      begin: The offset of its first byte in the file.
      end: The offset after its last byte.
      items: Its metadata, by key; each value as the file holds it.
  """

  index: int
  data_type: int
  begin: int
  end: int
  items: dict[str, object]

  @property
  def model_type(self) -> str | None:
    """Its `model_type` metadata, for a TFLite model; else None."""
    value = self.items.get('model_type')
    return value if isinstance(value, str) else None


def _value(pair: schema.KeyValuePair) -> object:
  """The value of a key-value pair of the header, as Python sees it."""
  table = pair.Value()
  kind = pair.ValueType()
  if table is None or kind == schema.VData.NONE:
    return None
  names = {
      value: name
      for name, value in vars(schema.VData).items()
      if isinstance(value, int)
  }
  holder = getattr(schema, names[kind])()
  holder.Init(table.Bytes, table.Pos)
  value = holder.Value()
  return value.decode() if isinstance(value, bytes) else value


def _key(pair: schema.KeyValuePair) -> str:
  return pair.Key().decode()


def _table(flatbuffer_object: Any) -> flatbuffers.table.Table:
  """Returns the raw table of a generated FlatBuffers object.

  The generated Python classes offer no public accessor for it.

  Args:
      flatbuffer_object: A generated FlatBuffers object.

  Returns:
      Its underlying table.
  """
  return flatbuffer_object._tab  # pylint: disable=protected-access


class LitertlmFile:
  """The header and sections of a .litertlm file mapped in memory.

  Attributes:
      data: The whole file, read-only.
      sections: Its sections, in the header's order.
      system: Its system metadata, by key.
  """

  def __init__(self, data: mmap.mmap | bytes) -> None:
    """Reads the header of a whole file.

    Args:
        data: The file's bytes.

    Raises:
        ValueError: It is not a .litertlm file this reads.
    """
    self.data = data
    if len(data) < HEADER_BEGIN or data[:8] != MAGIC:
      raise ValueError('This is not a .litertlm file.')
    (end,) = _UINT64.unpack_from(data, _HEADER_END_AT)
    if not HEADER_BEGIN < end <= len(data):
      raise ValueError('The .litertlm header ends outside the file.')
    # A copy: the header is small, and no view of the file outlives it.
    self._header = bytes(data[HEADER_BEGIN:end])
    metadata = schema.LiteRTLMMetaData.GetRootAs(self._header, 0)
    system = metadata.SystemMetadata()
    self.system = {}
    if system is not None:
      for i in range(system.EntriesLength()):
        entry = system.Entries(i)
        self.system[_key(entry)] = _value(entry)
    listed = metadata.SectionMetadata()
    self.sections = []
    for i in range(listed.ObjectsLength() if listed else 0):
      item = listed.Objects(i)
      items = {}
      for j in range(item.ItemsLength()):
        pair = item.Items(j)
        items[_key(pair)] = _value(pair)
      section = Section(
          index=i,
          data_type=item.DataType(),
          begin=item.BeginOffset(),
          end=item.EndOffset(),
          items=items,
      )
      if not end <= section.begin <= section.end <= len(data):
        raise ValueError(f'Section {i} of the .litertlm file is outside it.')
      self.sections.append(section)
    for before, after in zip(self.sections, self.sections[1:]):
      if after.begin < before.end:
        raise ValueError('The .litertlm sections overlap or are out of order.')

  def header(self) -> bytes:
    """The header's FlatBuffer bytes, as the file holds them."""
    return self._header

  def bytes_of(self, section: Section) -> 'memoryview[int]':
    """A section's bytes, without copying them.

    The view must be released before the file is unmapped.

    Args:
        section: A section of this file.

    Returns:
        A view of the section's bytes in the file.
    """
    return memoryview(self.data)[section.begin : section.end]

  def prefill_decode(self) -> Section:
    """The TFLite section of the text model's prefill and decode.

    Returns:
        The only section holding a prefill and decode TFLite model.

    Raises:
        ValueError: The file has none, or more than one.
    """
    found = [
        section
        for section in self.sections
        if section.data_type == schema.AnySectionDataType.TFLiteModel
        and section.model_type in PREFILL_DECODE
    ]
    if len(found) != 1:
      raise ValueError(
          'The model has no prefill and decode TFLite section of its '
          'own; only such a model can be observed.'
      )
    return found[0]

  def patched_header(
      self, offsets: dict[int, tuple[int, int]], uuid: str
  ) -> bytes:
    """The header with sections moved and another uuid, patched in place.

    Args:
        offsets: The new (begin, end) of each section that moves, by index.
        uuid: The system metadata's new `uuid`; as long as the old one.

    Returns:
        The header's FlatBuffer bytes.

    Raises:
        ValueError: A field to patch is not in the header, or the uuid's
          length differs.
    """
    header = bytearray(self.header())
    listed = schema.LiteRTLMMetaData.GetRootAs(header, 0).SectionMetadata()
    for index, (begin, end) in offsets.items():
      table = _table(listed.Objects(index))
      for slot, value in ((_BEGIN_SLOT, begin), (_END_SLOT, end)):
        field = table.Offset(slot)
        if not field:
          raise ValueError(
              f'Section {index} of the header has no offset to move.'
          )
        _UINT64.pack_into(header, table.Pos + field, value)
    system = schema.LiteRTLMMetaData.GetRootAs(header, 0).SystemMetadata()
    for i in range(system.EntriesLength() if system else 0):
      entry = system.Entries(i)
      if _key(entry) != 'uuid':
        continue
      if entry.ValueType() != schema.VData.StringValue:
        raise ValueError('The header has a uuid that is not a string.')
      table = entry.Value()
      holder = schema.StringValue()
      holder.Init(table.Bytes, table.Pos)
      field = _table(holder).Offset(_STRING_SLOT)
      start = _table(holder).Indirect(_table(holder).Pos + field)
      (length,) = struct.unpack_from('<I', header, start)
      new = uuid.encode()
      if length != len(new):
        raise ValueError(
            f'The header holds a uuid of another length than {len(new)} bytes.'
        )
      header[start + 4 : start + 4 + length] = new
    return bytes(header)


def round_up(offset: int) -> int:
  """`offset` rounded up to a multiple of BLOCK_BYTES."""
  return -(-offset // BLOCK_BYTES) * BLOCK_BYTES


@contextlib.contextmanager
def mapped(path: pathlib.Path) -> Iterator[LitertlmFile]:
  """Maps a .litertlm file read-only for a `with` block, and reads it.

  No view of its bytes may outlive the block.

  Args:
      path: The file.

  Yields:
      The file.

  Raises:
      ValueError: As LitertlmFile().
  """
  with open(path, 'rb') as file:
    data = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    try:
      yield LitertlmFile(data)
    finally:
      try:
        data.close()
      except BufferError:
        # A view of it is still alive, such as one an error's
        # traceback holds: the map closes once the last view goes.
        pass


def write_copy(
    source: LitertlmFile,
    replaced: Section,
    length: int,
    write_section: Callable[[BinaryIO], None],
    uuid: str,
    source_sha256: str,
    out: BinaryIO,
) -> None:
  """Writes the source with one section's bytes replaced.

  Every byte before the replaced section is the source's, the header
  patched (patched_header()); the replaced section keeps its begin; the
  bytes after it, the next section on, are the source's, moved by a whole
  number of blocks (layout()). Between the replaced section and the next
  are zeros: the source's bytes there must be zeros too, since they would
  have nowhere to go.

  The source is hashed as it is read, every byte of it in order, and must
  be the file `source_sha256` names.

  Args:
      source: The file copied.
      replaced: Its section whose bytes are replaced.
      length: How many bytes the replacement has.
      write_section: Writes them to the stream it is given, from where the
        section begins.
      uuid: The copy's `uuid`.
      source_sha256: SHA-256 of the source file, as the caller knows it.
      out: Where the copy goes, a new, empty file.

  Raises:
      ValueError: write_section wrote another length, the source holds
        bytes after the replaced section that a copy cannot place, the
        source is not the file `source_sha256` names, or the header cannot
        be patched.
  """
  offsets, shift = layout(source, replaced, length)
  after = _after(source, replaced)
  digest = hashlib.sha256()
  header = source.patched_header(offsets, uuid)
  head_end = HEADER_BEGIN + len(header)
  with memoryview(source.data) as data:
    digest.update(data[:head_end])
    out.write(data[:_HEADER_END_AT])
    out.write(_UINT64.pack(head_end))
    out.write(header)
    copy_bytes(data[head_end : replaced.begin], out, digest.update)
    for begin in range(replaced.begin, replaced.end, _CHUNK):
      digest.update(data[begin : min(begin + _CHUNK, replaced.end)])
    write_section(out)
    if out.tell() != replaced.begin + length:
      raise ValueError(
          f'The new section has {out.tell() - replaced.begin} bytes, '
          f'not {length}.'
      )
    padding = data[replaced.end : after]
    digest.update(padding)
    if not zeros(padding):
      raise ValueError(
          'The .litertlm file holds bytes after its section '
          f'{replaced.index} that a copy cannot place.'
      )
    if shift is not None:
      _pad_to(out, after + shift)
      copy_bytes(data[after:], out, digest.update)
    else:
      # The replaced section is the last: the file ends at the block
      # after it, or right after it, as the source does.
      end = replaced.begin + length
      _pad_to(out, round_up(end) if after > replaced.end else end)
  if digest.hexdigest() != source_sha256:
    raise ValueError(
        'The model file is not the one this server registered: its '
        'SHA-256 changed. Register it again.'
    )


def layout(
    source: LitertlmFile, replaced: Section, length: int
) -> tuple[dict[int, tuple[int, int]], int | None]:
  """Where the sections of a copy lie that move, as write_copy() writes it.

  Args:
      source: The file copied.
      replaced: Its section whose bytes are replaced.
      length: How many bytes the replacement has.

  Returns:
      The (begin, end) of the replaced section and of every one after it,
      by index; and how far those after it move, the least multiple of
      BLOCK_BYTES that clears the replacement (None when none comes after
      it).
  """
  offsets = {replaced.index: (replaced.begin, replaced.begin + length)}
  later = source.sections[replaced.index + 1 :]
  if not later:
    return offsets, None
  shift = round_up(replaced.begin + length - later[0].begin)
  for section in later:
    offsets[section.index] = (section.begin + shift, section.end + shift)
  return offsets, shift


def check_copy(source: LitertlmFile, copy: LitertlmFile, index: int) -> int:
  """Checks a copy write_copy() wrote, all but its replaced section's bytes.

  Its header must be the source's patched with the copy's offsets and uuid;
  every byte before the replaced section the source's; every byte from the
  next section on the source's, moved as layout() moves them; the bytes
  between the replaced section and the next zeros.

  Args:
      source: The file copied.
      copy: The copy.
      index: The replaced section's index.

  Returns:
      How many bytes outside the replaced section were compared.

  Raises:
      ValueError: The copy differs otherwise.
  """
  if len(copy.sections) != len(source.sections):
    raise ValueError('The copy has another number of sections.')
  replaced = source.sections[index]
  new = copy.sections[index]
  offsets, shift = layout(source, replaced, new.end - new.begin)
  for a, b in zip(source.sections, copy.sections):
    if (b.begin, b.end) != offsets.get(a.index, (a.begin, a.end)):
      raise ValueError(f'Section {a.index} of the copy is misplaced.')
  if copy.header() != source.patched_header(
      offsets, str(copy.system.get('uuid', ''))
  ):
    raise ValueError("The copy's header differs from the source's.")
  after = _after(source, replaced)
  head_end = HEADER_BEGIN + len(copy.header())
  with memoryview(source.data) as x, memoryview(copy.data) as y:
    if not same(x[:_HEADER_END_AT], y[:_HEADER_END_AT]):
      raise ValueError("The copy's first bytes differ from the source's.")
    _compare(x, y, head_end, replaced.begin, 0, source.sections[:index])
    if shift is None:
      end = len(y)
      if end not in (new.end, round_up(new.end)):
        raise ValueError('The copy does not end where it should.')
    else:
      end = after + shift
      if len(y) - end != len(x) - after:
        raise ValueError('The copy does not end where it should.')
      _compare(x, y, after, len(x), shift, source.sections[index + 1 :])
    if not zeros(y[new.end : end]):
      raise ValueError(
          f'The copy holds bytes after its section {index} that are not zeros.'
      )
  return len(source.data) - (replaced.end - replaced.begin)


def _compare(
    source: 'memoryview[int]',
    copy: 'memoryview[int]',
    begin: int,
    end: int,
    shift: int,
    sections: list[Section],
) -> None:
  """Compares source[begin:end] with copy moved by `shift`, section by section.

  Args:
      source: The source.
      copy: The copy.
      begin: Where the run of bytes compared begins in the source.
      end: Where it ends.
      shift: How far the copy moved it.
      sections: The source's sections in the run.

  Raises:
      ValueError: A section, or the bytes between, differ.
  """
  cursor = begin
  for section in [*sections, None]:
    stop = end if section is None else section.begin
    if not same(source[cursor:stop], copy[cursor + shift : stop + shift]):
      raise ValueError(
          "The bytes between the copy's sections differ from the source's."
      )
    if section is None:
      return
    if not same(
        source[section.begin : section.end],
        copy[section.begin + shift : section.end + shift],
    ):
      raise ValueError(f"Section {section.index} differs from the source's.")
    cursor = section.end


def _after(source: LitertlmFile, replaced: Section) -> int:
  """Where the next section after one begins, or the file's end."""
  later = source.sections[replaced.index + 1 :]
  return later[0].begin if later else len(source.data)


def copy_bytes(
    view: 'memoryview[int]',
    out: BinaryIO,
    update: Callable[['memoryview[int]'], object] | None = None,
) -> None:
  """Writes `view` to `out` a chunk at a time, each passed to `update`."""
  for begin in range(0, len(view), _CHUNK):
    chunk = view[begin : begin + _CHUNK]
    if update is not None:
      update(chunk)
    out.write(chunk)


def same(x: 'memoryview[int]', y: 'memoryview[int]') -> bool:
  """Whether two views hold the same bytes, compared a chunk at a time."""
  if len(x) != len(y):
    return False
  # As bytes: comparing memoryviews goes item by item, bytes by memcmp.
  return all(
      bytes(x[begin : begin + _CHUNK]) == bytes(y[begin : begin + _CHUNK])
      for begin in range(0, len(x), _CHUNK)
  )


def zeros(view: 'memoryview[int]') -> bool:
  """Whether a view holds zeros alone."""
  return all(
      not bytes(view[begin : begin + _CHUNK]).strip(b'\0')
      for begin in range(0, len(view), _CHUNK)
  )


def _pad_to(out: BinaryIO, offset: int) -> None:
  """Writes zeros up to `offset`."""
  if out.tell() > offset:
    raise ValueError('The copy overran a section of the .litertlm file.')
  out.write(bytes(offset - out.tell()))
