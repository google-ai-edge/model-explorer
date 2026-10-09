#!/usr/bin/env python3
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

"""Validate an exported Runner folder and index its original Safetensors."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

from model_explorer_debugger.runtime import litert_capture
from model_explorer_debugger.runtime import litert_dump_inference
from model_explorer_debugger.runtime import litert_evidence_model
from model_explorer_debugger.runtime import litert_lm_adapter
from model_explorer_debugger.runtime import litert_trace


def digest(path):
  with path.open('rb') as file:
    return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--runtime-root', type=Path, required=True)
  parser.add_argument('--capture', type=Path, required=True)
  parser.add_argument(
      '--model',
      type=Path,
      required=True,
      help='Exact prepared model retained on the Mac',
  )
  parser.add_argument(
      '--output', type=Path, required=True, help='New output directory'
  )
  args = parser.parse_args()
  source = args.capture.resolve()
  job = json.loads((source / 'job.json').read_text())
  result = json.loads((source / 'result.json').read_text())
  if args.output.exists():
    parser.error('Output must be a new directory')
  if (source / 'failure.json').exists() or not result.get('tensors'):
    raise ValueError('Capture is incomplete or empty')
  if (
      result['jobID'].lower() != job['id'].lower()
      or result['modelSHA256'] != job['modelSHA256']
      or digest(args.model) != job['modelSHA256']
  ):
    raise ValueError('Model/job identity mismatch')
  backend = job.get('backend', 'CPU')
  evidence = result.get('backendEvidence') or {}
  if (
      backend not in ('CPU', 'GPU')
      or result['backend'] != backend
      or result['debuggerEnabled'] is not True
      or (
          backend == 'GPU'
          and (
              evidence.get('engineInitialized') is not True
              or evidence.get('effectiveBackend') != backend
          )
      )
  ):
    raise ValueError('Unsupported capture provenance')
  package_root = Path(__file__).resolve().parents[2]
  sys.path[:0] = [str(package_root), str(args.runtime_root / 'src')]

  args.output.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(
      dir=args.output.parent, prefix='.ios-import-'
  ) as temp:
    staging = Path(temp)
    manifest = staging / 'manifest.json'
    manifest.write_text(json.dumps(job['manifest']))
    verified = litert_lm_adapter.verify_taps(args.model, manifest)
    manifest.write_text(json.dumps(verified))
    native_trace = source / 'raw/runtime_trace.jsonl'
    # Without a native trace, a dump that carries Decode logits still supports
    # inferred generation evidence, so the whole raw directory is retained
    # either way.
    full_raw = native_trace.is_file() or litert_dump_inference.has_dump_logits(
        source / 'raw'
    )
    tensors = staging / 'raw'
    tensors.mkdir()
    seen = set()
    for record in result['tensors']:
      path = (source / record['path']).resolve()
      if (
          not path.is_relative_to(source / 'tensors')
          or path.suffix != '.safetensors'
          or path.name in seen
      ):
        raise ValueError('Unsafe or duplicate capture path')
      seen.add(path.name)
      if digest(path) != record['sha256']:
        raise ValueError('Captured file checksum mismatch')
      if full_raw:
        raw = source / 'raw' / path.name
        if not raw.is_file() or digest(raw) != record['sha256']:
          raise ValueError('Native raw capture differs from selected tensor')
      else:
        shutil.copy2(path, tensors / path.name)
    if full_raw:
      for path in (source / 'raw').iterdir():
        if (
            not path.is_file()
            or path.is_symlink()
            or path.resolve().parent != source / 'raw'
            or (
                path.suffix != '.safetensors'
                and path.name
                not in ('runtime_trace.jsonl', 'generated_tokens.jsonl')
            )
        ):
          raise ValueError('Invalid native raw evidence file')
        shutil.copy2(path, tensors / path.name)
    exported = staging / 'export'
    index = litert_capture.collect(
        tensors, manifest, exported, backend, reference=True
    )
    # Treat phone metadata as claims; validate it against actual tensors.
    declared = {
        (r['signature'], r['key'], r['step'], tuple(r['shape']))
        for r in result['tensors']
    }
    actual = {
        (r['signature'], r['key'], r['step'], tuple(r['shape']))
        for r in index['tensors']
    }
    if declared != actual:
      raise ValueError('Result metadata differs from tensor payloads')
    dtype_names = {
        'F32': 'float32',
        'F16': 'float16',
        'I32': 'int32',
        'U8': 'uint8',
        'I64': 'int64',
        'BOOL': 'bool',
        'I8': 'int8',
    }
    claimed_dtypes = {
        (r['signature'], r['key'], r['step']): dtype_names.get(r['dtype'])
        for r in result['tensors']
    }
    for record in index['tensors']:
      if (
          claimed_dtypes[(record['signature'], record['key'], record['step'])]
          != record['dtype']
      ):
        raise ValueError('Declared dtype differs from captured dtype')
    if native_trace.is_file():
      litert_trace.attach_trace(
          index,
          tensors,
          verified,
          litert_evidence_model.describe_model(args.model, verified),
          turn=result.get('turnSequence', 0),
      )
    elif full_raw:
      # KV snapshots need the model's cache descriptors; Token Diff evidence
      # must not depend on them.
      model_info, model_reason = None, None
      try:
        model_info = litert_evidence_model.describe_model(args.model, verified)
      except Exception as error:  # noqa: BLE001
        # Recorded as the reason KV stays unindexed.
        model_reason = f'model KV cache descriptors unavailable: {error}'
      build_file = source / 'runtime-build.json'
      build = (
          json.loads(build_file.read_text()) if build_file.is_file() else None
      )
      litert_dump_inference.infer_generation(
          index,
          tensors,
          result,
          job,
          turn=result.get('turnSequence', 0),
          model=model_info,
          model_reason=model_reason,
          build=build,
      )
    if full_raw:
      # Token Diff labels any vocabulary ID from the model's own tokenizer,
      # saved with the capture.
      index['tokenizer'] = litert_evidence_model.export_tokenizer(
          args.model, exported / 'tokenizer.model'
      )
    shutil.move(str(tensors), exported / 'raw')
    index['tensor_root'] = 'export'
    for record in index['tensors']:
      record['path'] = str(Path('raw') / Path(record['source']).name)
      record['source'] = str(args.output.resolve() / record['path'])
    (exported / 'capture_index.json').write_text(
        json.dumps(index, indent=2) + '\n'
    )
    shutil.copy2(source / 'job.json', exported / 'job.json')
    shutil.copy2(source / 'result.json', exported / 'runner-result.json')
    if (source / 'runtime-build.json').is_file():
      shutil.copy2(
          source / 'runtime-build.json', exported / 'runtime-build.json'
      )
    exported.rename(args.output)
  print(
      f"Validated {len(index['tensors'])} tensors:"
      f" {args.output.resolve() / 'capture_index.json'}"
  )


if __name__ == '__main__':
  main()
