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

"""Disposable, cancellable model preparation worker.

No native inference is required.
"""

import hashlib
import json
import mmap
from pathlib import Path
import shutil
import sys
import tempfile

from ..fsutil import atomic_json, file_digest
from .litert_lm_adapter import verify_taps
from .tap_profiles import CUSTOM_PROFILE, PROFILE_ID, expected_manifest


def verify_buffers(source, destination):
  """Compare all embedded/offset buffer payloads after FlatBuffer relocation."""
  from debugger_tap import tflite_generated as fb

  with source.open('rb') as left, destination.open('rb') as right:
    with (
        mmap.mmap(left.fileno(), 0, access=mmap.ACCESS_READ) as a,
        mmap.mmap(right.fileno(), 0, access=mmap.ACCESS_READ) as b,
    ):
      models = [fb.Model.GetRootAs(data, 0) for data in (a, b)]
      if models[0].BuffersLength() != models[1].BuffersLength():
        raise ValueError('Buffer count changed during tap preparation')

      def checksum(data, buf):
        if buf.Offset() > 1:
          start, size = buf.Offset(), buf.Size()
        else:
          offset = buf._tab.Offset(4)
          start, size = (
              (buf._tab.Vector(offset), buf.DataLength()) if offset else (0, 0)
          )
        if start + size > len(data):
          raise ValueError('Invalid buffer extent')
        digest = hashlib.sha256()
        for position in range(start, start + size, 8 << 20):
          digest.update(
              data[position : min(position + (8 << 20), start + size)]
          )
        return size, digest.digest()

      for i in range(models[0].BuffersLength()):
        if checksum(a, models[0].Buffers(i)) != checksum(
            b, models[1].Buffers(i)
        ):
          raise ValueError(f'Buffer {i} changed during tap preparation')
      return {
          'buffers_checked': models[0].BuffersLength(),
          'all_payloads_identical': True,
      }


def prepare(request, emit):
  root = Path(request['runtime_root'])
  sys.path.insert(0, str(root / 'src'))
  from debugger_tap.tap import rewrite
  from litert_lm_builder import litertlm_builder as builder

  if request['tap_profile'] not in (PROFILE_ID, CUSTOM_PROFILE):
    raise ValueError('Unknown tensor capture profile')
  source = Path(request['model'])
  emit('Checking source model')
  if file_digest(source) != request['model_sha256']:
    raise ValueError('Registered model changed; upload or register it again')
  expected = expected_manifest()
  custom = request['tap_profile'] == CUSTOM_PROFILE
  selection = request.get('tap_selection')
  if custom and (not selection or not 1 <= len(selection['points']) <= 16):
    raise ValueError('Select 1–16 capture points before preparing')
  selection_key = (
      hashlib.sha256(
          json.dumps(selection, sort_keys=True).encode()
      ).hexdigest()[:24]
      if custom
      else ''
  )
  cache = Path(request['tap_cache']) / (
      request['model_sha256'] + '-' + request['tap_profile'] + selection_key
  )
  cache.parent.mkdir(parents=True, exist_ok=True)
  if (cache / 'result.json').is_file():
    result = json.loads((cache / 'result.json').read_text())
    if file_digest(Path(result['model'])) != result['model_sha256']:
      raise ValueError('Prepared model cache failed its checksum verification')
    verify_taps(result['model'], result['manifest'])
    emit('Reusing verified debug model')
    return {**result, 'reused': True}
  if shutil.disk_usage(cache.parent).free < source.stat().st_size * 4 + 1024**3:
    raise ValueError(
        'Not enough free disk space to prepare and verify this model (about'
        ' four model copies required)'
    )
  # On failure a partial directory is never a valid cache entry. SIGKILL leaves
  # only a .pending directory, never a registered model or updated session.
  with tempfile.TemporaryDirectory(
      prefix='.pending-', dir=Path(request['output'])
  ) as temporary:
    work = Path(temporary)
    emit('Unpacking model')
    toml = Path(builder.unpack(str(source), str(work / 'unpacked')))
    import tomllib

    raw = toml.read_text()
    sections = tomllib.loads(raw)['section']
    matches = [
        s
        for s in sections
        if s.get('section_type') == 'TFLiteModel'
        and s.get('model_type', '').lower() == 'prefill_decode'
    ]
    if len(matches) != 1:
      raise ValueError('Expected one prefill/decode model section')
    section = toml.parent / matches[0]['data_path']
    checksum = file_digest(section)
    manifest_path = work / 'tap-manifest.json'
    if not custom and checksum == expected['tapped_sha256']:
      emit('Verifying existing taps')
      atomic_json(manifest_path, expected)
      verify_taps(source, manifest_path)
      # Keep the existing immutable model; only attach the reviewed profile.
      model_path = source
      verification = {'already_tapped': True}
    elif checksum == (
        selection['section_sha256'] if custom else expected['source_sha256']
    ):
      emit(
          'Adding selected tensor outputs'
          if custom
          else 'Adding Layer 0 RMSNorm outputs'
      )
      tapped = work / 'prefill-decode.tflite'
      manifest = rewrite(
          section,
          tapped,
          [
              {k: t[k] for k in ('signature', 'op', 'output')}
              for t in (selection['points'] if custom else expected['taps'])
          ],
      )
      fields = (
          'signature',
          'subgraph',
          'op',
          'output',
          'tensor',
          'tensor_name',
          'tensor_type',
          'shape',
      )
      if custom and any(
          any(a[k] != b[k] for k in fields)
          for a, b in zip(manifest['taps'], selection['points'])
      ):
        raise ValueError('Selected output no longer matches the scanned model')
      if not custom and manifest['taps'] != expected['taps']:
        raise ValueError(
            'Rewritten tap coordinates differ from the reviewed profile'
        )
      atomic_json(manifest_path, manifest)
      emit('Verifying weight buffers')
      verification = verify_buffers(section, tapped)
      for item in sections:
        path = item.get('data_path')
        if path is None:
          continue
        replacement = tapped if item is matches[0] else toml.parent / path
        needle = 'data_path = ' + json.dumps(path)
        if raw.count(needle) != 1:
          raise ValueError('Ambiguous container section path')
        raw = raw.replace(
            needle, 'data_path = ' + json.dumps(str(replacement.resolve()))
        )
      config = work / 'repack.toml'
      config.write_text(raw)
      emit('Packing debug model')
      model_path = work / (
          'custom-tapped.litertlm'
          if custom
          else 'gemma4-e2b-layer0-tapped.litertlm'
      )
      builder.pack(str(config), str(model_path))
      verify_taps(model_path, manifest_path)
    else:
      raise ValueError(
          'This model build is not supported by the Gemma 4 E2B Layer 0'
          ' profile. Select the reviewed E2B model or disable intermediate'
          ' tensor capture.'
      )
    emit('Publishing verified debug model')
    output = work / 'published'
    output.mkdir()
    shutil.copyfile(manifest_path, output / 'tap-manifest.json')
    if model_path.parent == work:
      model_path.rename(output / model_path.name)
      final_model = cache / model_path.name
      model_hash = file_digest(output / model_path.name)
    else:
      final_model, model_hash = model_path, request['model_sha256']
    result = dict(
        model=str(final_model),
        model_sha256=model_hash,
        manifest=str(cache / 'tap-manifest.json'),
        profile=request['tap_profile'],
        reviewed_e2b=selection.get('reviewed_e2b', False) if custom else True,
        source_sha256=request['model_sha256'],
        verification=verification,
        reused=False,
    )
    atomic_json(output / 'result.json', result)
    output.rename(cache)
    return result


def main():
  request = json.loads(Path(sys.argv[1]).read_text())
  output = Path(request['output'])
  output.mkdir(parents=True, exist_ok=True)

  def event(**value):
    if (output.parent / 'cancel').exists():
      raise InterruptedError('Cancelled')
    with (output / 'events.jsonl').open('a') as stream:
      stream.write(json.dumps(value) + '\n')

  try:
    result = prepare(
        request, lambda message: event(type='progress', message=message)
    )
    atomic_json(output / 'result.json', result)
  except Exception as error:
    event(type='error', error=str(error))
    raise


if __name__ == '__main__':
  main()
