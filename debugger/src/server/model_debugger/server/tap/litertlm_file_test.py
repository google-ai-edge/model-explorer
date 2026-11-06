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

"""Tests of reading a .litertlm file and copying it with a section replaced."""

import hashlib
import pathlib
import struct

from absl.testing import absltest
import flatbuffers
from model_debugger.server.tap import litertlm_file
from model_debugger.server.tap import litertlm_schema_generated as schema
from model_debugger.server.testing import tap_fixtures
from model_debugger.server.testing import test_main

_UUID = '11111111-2222-5333-8444-555555555555'


def _sha256(path: pathlib.Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class LitertlmFileTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp_path = pathlib.Path(self.create_tempdir().full_path)
    self.path = self.tmp_path / 'model.litertlm'
    tap_fixtures.write_model(self.path)

  def _copy(
      self, replacement: bytes, sha256: str | None = None
  ) -> pathlib.Path:
    """The fixture with its text model replaced, written and checked."""
    copy = self.tmp_path / 'copy.litertlm'
    copy.unlink(missing_ok=True)
    with litertlm_file.mapped(self.path) as source:
      with open(copy, 'xb') as out:
        litertlm_file.write_copy(
            source,
            source.prefill_decode(),
            len(replacement),
            lambda stream: stream.write(replacement),
            _UUID,
            sha256 or _sha256(self.path),
            out,
        )
      with litertlm_file.mapped(copy) as written:
        litertlm_file.check_copy(source, written, 1)
    return copy

  def test_reads_the_sections_and_system_metadata(self):
    """Each section's type, range and metadata, as written."""
    with litertlm_file.mapped(self.path) as model:
      self.assertEqual(
          [section.data_type for section in model.sections],
          [
              schema.AnySectionDataType.LlmMetadataProto,
              schema.AnySectionDataType.TFLiteModel,
              schema.AnySectionDataType.TFLiteModel,
          ],
      )
      self.assertEqual(
          model.system,
          {
              'author': 'The tests',
              'uuid': tap_fixtures.UUID,
              'creation_timestamp': '2026-10-18T00:00:00+00:00',
          },
      )
      text = model.prefill_decode()
      self.assertEqual(text.index, 1)
      self.assertEqual(text.model_type, 'tf_lite_prefill_decode')
      with model.bytes_of(text) as data:
        self.assertEqual(bytes(data), tap_fixtures.text_model())

  def test_refuses_what_is_not_a_litertlm_file(self):
    """Another file, or one without a text model, is refused."""
    other = self.tmp_path / 'other.litertlm'
    other.write_bytes(b'TFL3' * 100)
    with self.assertRaisesRegex(ValueError, 'not a .litertlm'):
      with litertlm_file.mapped(other):
        pass
    with litertlm_file.mapped(self.path) as model:
      model.sections.pop(1)
      with self.assertRaisesRegex(ValueError, 'no prefill and decode'):
        model.prefill_decode()

  def test_a_copy_keeps_every_byte_but_the_section_and_uuid(self):
    """The other sections and metadata are the source's; later ones move."""
    replacement = b'r' * 20000
    copy = self._copy(replacement)
    with litertlm_file.mapped(self.path) as source:
      before = [
          (section, bytes(source.bytes_of(section)))
          for section in source.sections
      ]
      header = source.header()
    with litertlm_file.mapped(copy) as written:
      self.assertEqual(written.system, {**source.system, 'uuid': _UUID})
      after = written.sections
      self.assertEqual(after[0], before[0][0])
      self.assertEqual(after[1].begin, before[1][0].begin)
      self.assertLen(replacement, after[1].end - after[1].begin)
      self.assertEqual(after[2].begin % litertlm_file.BLOCK_BYTES, 0)
      self.assertGreater(after[2].begin, before[2][0].begin)
      self.assertEqual(after[2].items, before[2][0].items)
      self.assertEqual(bytes(written.bytes_of(after[0])), before[0][1])
      self.assertEqual(bytes(written.bytes_of(after[1])), replacement)
      self.assertEqual(bytes(written.bytes_of(after[2])), before[2][1])
      # The header is the source's, patched in place: same length.
      self.assertLen(written.header(), len(header))
    self.assertEqual(copy.stat().st_size % litertlm_file.BLOCK_BYTES, 0)

  def test_a_copy_refuses_a_section_of_another_length(self):
    """write_section must write the length announced."""
    copy = self.tmp_path / 'copy.litertlm'
    with litertlm_file.mapped(self.path) as source:
      with open(copy, 'xb') as out:
        with self.assertRaisesRegex(ValueError, 'not 10'):
          litertlm_file.write_copy(
              source,
              source.prefill_decode(),
              10,
              lambda stream: stream.write(b'x' * 11),
              _UUID,
              _sha256(self.path),
              out,
          )

  def test_a_copy_keeps_the_bytes_between_and_after_sections(self):
    """Moved with the sections after the replaced one, kept before it."""
    tap_fixtures.write_model(self.path, between=b'not zeros')
    for replacement in (b'r' * 40000, b's' * 10):
      with self.subTest(length=len(replacement)):
        copy = self._copy(replacement)
        self.assertEqual(copy.read_bytes().count(b'not zeros'), 3)
        with litertlm_file.mapped(self.path) as source:
          with litertlm_file.mapped(copy) as written:
            shift = written.sections[2].begin - source.sections[2].begin
            self.assertEqual(shift % litertlm_file.BLOCK_BYTES, 0)
            self.assertEqual(
                bytes(written.data[written.sections[2].begin :]),
                bytes(source.data[source.sections[2].begin :]),
            )
            head = 32 + len(source.header())
            self.assertEqual(
                bytes(written.data[head : source.sections[1].begin]),
                bytes(source.data[head : source.sections[1].begin]),
            )

  def test_a_copy_refuses_bytes_it_cannot_place(self):
    """Bytes right after the replaced section have nowhere to go."""
    tap_fixtures.write_model(self.path, after_text=b'no place')
    with self.assertRaisesRegex(ValueError, 'cannot place'):
      self._copy(b'r' * 40000)

  def test_a_copy_of_a_source_that_changed_is_refused(self):
    """Every byte read is hashed; the source must be the file named."""
    with self.assertRaisesRegex(ValueError, 'SHA-256 changed'):
      self._copy(b'r' * 100, sha256='0' * 64)

  def test_the_check_finds_a_change_outside_the_section(self):
    """The header, the bytes between sections, the zeros after it."""
    tap_fixtures.write_model(self.path, between=b'not zeros')
    good = self._copy(b'r' * 40000).read_bytes()
    for found, what in (
        (good.index(b'The tests'), 'header differs'),
        (good.index(b'not zeros'), 'bytes between'),
        (good.rindex(b'not zeros'), 'bytes between'),
        (good.index(b'r' * 40000) + 40001, 'not zeros'),
        (8, 'first bytes'),
    ):
      with self.subTest(what=what):
        self.assertGreater(found, 0)
        changed = bytearray(good)
        changed[found] ^= 1
        copy = self.tmp_path / 'changed.litertlm'
        copy.write_bytes(changed)
        with litertlm_file.mapped(self.path) as source:
          with litertlm_file.mapped(copy) as written:
            with self.assertRaisesRegex(ValueError, what):
              litertlm_file.check_copy(source, written, 1)

  def test_a_uuid_of_another_length_is_refused(self):
    """The uuid is patched in place, so it keeps its length."""
    with litertlm_file.mapped(self.path) as source:
      with self.assertRaisesRegex(ValueError, 'uuid of another length'):
        source.patched_header({}, 'short')

  def test_reads_empty_metadata_values_and_replaces_last_section(self):
    """NONE metadata pairs read as None; replacing the last section pads end."""
    pair = schema.KeyValuePairT()
    pair.key = 'unset'
    pair.valueType = schema.VData.NONE
    metadata = schema.LiteRTLMMetaDataT()
    metadata.systemMetadata = schema.SystemMetadataT()
    metadata.systemMetadata.entries = [pair]
    builder = flatbuffers.Builder(256)
    builder.Finish(metadata.Pack(builder))
    header = bytes(builder.Output())
    raw = (
        b'LITERTLM'
        + struct.pack('<III', 1, 5, 0)
        + bytes(4)
        + struct.pack('<Q', 32 + len(header))
        + header
    )
    self.assertEqual(litertlm_file.LitertlmFile(raw).system, {'unset': None})
    copy = self.tmp_path / 'last_replaced.litertlm'
    replacement = b'e' * 512
    with litertlm_file.mapped(self.path) as source:
      with open(copy, 'xb') as out:
        litertlm_file.write_copy(
            source,
            source.sections[-1],
            len(replacement),
            lambda stream: stream.write(replacement),
            _UUID,
            _sha256(self.path),
            out,
        )
      with litertlm_file.mapped(copy) as written:
        litertlm_file.check_copy(source, written, len(source.sections) - 1)
        self.assertEqual(
            bytes(written.bytes_of(written.sections[-1])), replacement
        )


if __name__ == '__main__':
  test_main.main()
