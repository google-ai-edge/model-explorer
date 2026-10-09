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

"""Read native KV ownership and tokenizer identity from the model container.

Both are read from the exact model container.
"""

import hashlib
import io
import mmap
from pathlib import Path
import re


def export_tokenizer(path, destination):
  """Copy the embedded SentencePiece tokenizer bytes to `destination`.

  The exact embedded bytes are copied, and the result describes them.
  """
  from litert_lm_builder.litertlm_peek import read_litertlm_header, schema

  metadata = read_litertlm_header(str(path), io.StringIO()).SectionMetadata()
  sections = [metadata.Objects(i) for i in range(metadata.ObjectsLength())]
  tokenizers = [
      section
      for section in sections
      if section.DataType() == schema.AnySectionDataType.SP_Tokenizer
  ]
  if len(tokenizers) != 1:
    raise ValueError('Missing or ambiguous native tokenizer section')
  start, end = tokenizers[0].BeginOffset(), tokenizers[0].EndOffset()
  if not 0 <= start < end <= Path(path).stat().st_size:
    raise ValueError('Invalid native tokenizer section')
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream, Path(destination).open('wb') as out:
    stream.seek(start)
    remaining = end - start
    while remaining:
      block = stream.read(min(8 << 20, remaining))
      if not block:
        raise ValueError('Native tokenizer section is truncated')
      out.write(block)
      digest.update(block)
      remaining -= len(block)
  return dict(
      kind='sentencepiece',
      path=Path(destination).name,
      sha256=digest.hexdigest(),
      size=end - start,
      section_offset=start,
      digest_scope='exact embedded SentencePiece tokenizer bytes',
  )


def describe_model(path, manifest):
  from debugger_tap import tflite_generated as fb
  from flatbuffers import flexbuffers
  from litert_lm_builder.litertlm_peek import read_litertlm_header, schema

  metadata = read_litertlm_header(str(path), io.StringIO()).SectionMetadata()
  sections = [metadata.Objects(i) for i in range(metadata.ObjectsLength())]
  offsets = {tap['section_offset'] for tap in manifest['taps']}
  if len(offsets) != 1:
    raise ValueError('Native evidence requires one verified text model section')
  section = next(
      (section for section in sections if section.BeginOffset() in offsets),
      None,
  )
  tokenizers = [
      section
      for section in sections
      if section.DataType() == schema.AnySectionDataType.SP_Tokenizer
  ]
  if section is None or len(tokenizers) != 1:
    raise ValueError('Missing or ambiguous native model/tokenizer section')
  with (
      Path(path).open('rb') as stream,
      mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data,
  ):

    def digest_section(item):
      start, end = item.BeginOffset(), item.EndOffset()
      if not 0 <= start < end <= len(data):
        raise ValueError('Invalid native model section')
      digest = hashlib.sha256()
      for offset in range(start, end, 8 << 20):
        digest.update(data[offset : min(offset + (8 << 20), end)])
      return digest.hexdigest()

    tokenizer_hash = digest_section(tokenizers[0])
    section_hash = digest_section(section)
    model = fb.Model.GetRootAs(data, section.BeginOffset())
    descriptors = {}
    for i in range(model.SignatureDefsLength()):
      signature = model.SignatureDefs(i)
      name = signature.SignatureKey().decode()
      graph = model.Subgraphs(signature.SubgraphIndex())
      producers = {}
      for op_index in range(graph.OperatorsLength()):
        op = graph.Operators(op_index)
        for output in range(op.OutputsLength()):
          producers.setdefault(op.Outputs(output), []).append((op_index, op))
      for output in range(signature.OutputsLength()):
        item = signature.Outputs(output)
        cache_name = item.Name().decode()
        match = re.fullmatch(r'kv_cache_([kv])_(\d+)', cache_name)
        if not match:
          continue
        tensor = graph.Tensors(item.TensorIndex())
        candidates = producers.get(item.TensorIndex(), [])
        if len(candidates) != 1:
          continue
        op_index, op = candidates[0]
        composite = fb.OperatorT.InitFromObj(op).builtinOptions2
        if (
            getattr(composite, 'name', None) != b'odml.cache_update'
            or op.InputsLength() < 2
        ):
          continue
        attributes = flexbuffers.Loads(bytes(composite.compositeAttributes))
        # The generated object may retain NumPy views of the mmap.
        del composite
        owner_inputs = [
            graph.Tensors(op.Inputs(j)).Name().decode() for j in (0, 1)
        ]
        owners = {
            int(found.group(1))
            for value in owner_inputs
            for found in re.finditer(r'/layer_(\d+)/', value)
        }
        if len(owners) != 1:
          continue
        kind = 'key' if match.group(1) == 'k' else 'value'
        layout = (
            ['batch', 'kv_head', 'sequence', 'head_dim']
            if kind == 'key'
            else ['batch', 'kv_head', 'head_dim', 'sequence']
        )
        shape = [tensor.Shape(j) for j in range(tensor.ShapeLength())]
        if (
            len(shape) != 4
            or shape[layout.index('sequence')] != attributes.get('cache_size')
            or shape[layout.index('head_dim')] != attributes.get('head_size')
        ):
          continue
        quant = tensor.Quantization()
        quantization = None
        if (
            tensor.Type() == fb.TensorType.INT8
            and quant
            and quant.ScaleLength() == quant.ZeroPointLength() == 1
        ):
          quantization = dict(
              scale=quant.Scale(0),
              zero_point=quant.ZeroPoint(0),
              quantized_dimension=quant.QuantizedDimension(),
          )
        descriptors[name, cache_name] = dict(
            layer=owners.pop(),
            kind=kind,
            slot=int(match.group(2)),
            layout=layout,
            model_shape=shape,
            quantization=quantization,
            source=dict(
                section_offset=section.BeginOffset(),
                section_sha256=section_hash,
                signature=name,
                subgraph=signature.SubgraphIndex(),
                op=op_index,
                output_tensor=item.TensorIndex(),
                composite='odml.cache_update',
                owner_input_tensor_names=owner_inputs,
                composite_attributes=attributes,
            ),
        )
  return dict(
      kv=descriptors,
      tokenizer=dict(
          vocab_sha256=tokenizer_hash,
          digest_scope='exact embedded SentencePiece tokenizer bytes',
          section_offset=tokenizers[0].BeginOffset(),
      ),
      section_sha256=section_hash,
  )
