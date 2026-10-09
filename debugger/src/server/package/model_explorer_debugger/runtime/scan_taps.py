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

"""Read the text model's concrete output identities.

Weights are never unpacked.
"""

import io
import json
import mmap
from pathlib import Path
import sys

from ..fsutil import atomic_json, file_digest
from .tap_profiles import expected_manifest


def scan(path, checksum, runtime_root):
  sys.path.insert(0, str(Path(runtime_root) / 'src'))
  from debugger_tap import tflite_generated as fb
  from litert_lm_builder.litertlm_peek import (
      read_litertlm_header,
      get_model_type,
  )

  if file_digest(Path(path)) != checksum:
    raise ValueError(
        'Model changed since registration; select a newly uploaded copy'
    )
  metadata = read_litertlm_header(str(path), io.StringIO()).SectionMetadata()
  matches = (
      [
          metadata.Objects(i)
          for i in range(metadata.ObjectsLength())
          if get_model_type(metadata.Objects(i))
          in ('prefill_decode', 'tf_lite_prefill_decode')
      ]
      if metadata
      else []
  )
  if len(matches) != 1:
    raise ValueError(
        'Custom capture currently requires one prefill/decode TFLite section'
    )
  section = matches[0]
  with (
      Path(path).open('rb') as stream,
      mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data,
  ):
    begin, end = section.BeginOffset(), section.EndOffset()
    if (
        not 0 <= begin < end <= len(data)
        or data[begin + 4 : begin + 8] != b'TFL3'
    ):
      raise ValueError('Invalid prefill/decode section')
    import hashlib

    hasher = hashlib.sha256()
    for position in range(begin, end, 8 << 20):
      hasher.update(data[position : min(position + (8 << 20), end)])
    section_hash = hasher.hexdigest()
    model = fb.Model.GetRootAs(data, begin)
    types = {
        value: key
        for key, value in vars(fb.TensorType).items()
        if isinstance(value, int)
    }
    codes = {
        value: key
        for key, value in vars(fb.BuiltinOperator).items()
        if isinstance(value, int)
    }
    signatures = [
        model.SignatureDefs(i) for i in range(model.SignatureDefsLength())
    ]
    aliases = {}
    for sig in signatures:
      aliases[sig.SubgraphIndex()] = aliases.get(sig.SubgraphIndex(), 0) + 1
    rows = []
    for sig in signatures:
      signature = sig.SignatureKey().decode()
      graph = model.Subgraphs(sig.SubgraphIndex())
      producers = {}
      for op_index in range(graph.OperatorsLength()):
        op = graph.Operators(op_index)
        for slot in range(op.OutputsLength()):
          tensor = op.Outputs(slot)
          producers[tensor] = producers.get(tensor, 0) + 1
      for op_index in range(graph.OperatorsLength()):
        op = graph.Operators(op_index)
        code = model.OperatorCodes(op.OpcodeIndex())
        builtin = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
        for slot in range(op.OutputsLength()):
          index = op.Outputs(slot)
          if index < 0:
            continue
          tensor = graph.Tensors(index)
          shape = [tensor.Shape(i) for i in range(tensor.ShapeLength())]
          dynamic = any(
              tensor.ShapeSignature(i) < 0
              for i in range(tensor.ShapeSignatureLength())
          )
          reason = ''
          if aliases[sig.SubgraphIndex()] != 1:
            reason = (
                'Shared signature subgraphs are not supported for custom'
                ' selection yet'
            )
          elif producers[index] != 1:
            reason = 'Tensor has multiple producers'
          elif tensor.IsVariable():
            reason = 'Variable tensor'
          elif tensor.Type() not in (0, 1, 2, 3, 4, 6, 9):
            reason = 'Unsupported capture dtype'
          elif not shape or any(d <= 0 for d in shape) or dynamic:
            reason = 'Dynamic or empty shape'
          elif builtin in (
              fb.BuiltinOperator.IF,
              fb.BuiltinOperator.WHILE,
              fb.BuiltinOperator.CALL_ONCE,
          ):
            reason = 'Control-flow output requires separate validation'
          rows.append(
              dict(
                  id=f'{signature}:{op_index}:{slot}',
                  signature=signature,
                  subgraph=sig.SubgraphIndex(),
                  op=op_index,
                  output=slot,
                  tensor=index,
                  tensor_name=(tensor.Name() or b'').decode(),
                  shape=shape,
                  tensor_type=tensor.Type(),
                  dtype=types.get(tensor.Type(), str(tensor.Type())),
                  operator=codes.get(builtin, str(builtin)),
                  selectable=not reason,
                  reason=reason,
              )
          )
    expected = expected_manifest()
    recommended = []
    if section_hash in (expected['source_sha256'], expected['tapped_sha256']):
      recommended = [
          f"{t['signature']}:{t['op']}:{t['output']}" for t in expected['taps']
      ]
    return dict(
        status='completed',
        model_sha256=checksum,
        section_sha256=section_hash,
        points=rows,
        recommended=recommended,
        scope='prefill_decode',
    )


if __name__ == '__main__':
  request = json.loads(Path(sys.argv[1]).read_text())
  try:
    result = scan(
        request['model'], request['model_sha256'], request['runtime_root']
    )
  except Exception as error:
    result = {'status': 'failed', 'error': str(error)}
  atomic_json(Path(request['result']), result)
