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

"""A tiny Gemma 4 with random weights, as a Hugging Face model directory.

Six text layers, the last two sharing the KV of earlier ones, sliding windows
of 4 positions, no vision or audio tower; a tokenizer of one token per
character, with Gemma 4's special tokens and a chat template of its shape.
The weights come from a fixed seed, so the model is the same every time.
"""

import pathlib

import tokenizers
from tokenizers import decoders
from tokenizers import models
from tokenizers import pre_tokenizers
import torch
import transformers

PAD, EOS, BOS, TURN_START, TURN_END = 0, 1, 2, 3, 4
SLIDING_WINDOW = 4
LAYER_TYPES = [
    'sliding_attention',
    'sliding_attention',
    'full_attention',
    'sliding_attention',
    'sliding_attention',
    'full_attention',
]
# The layers with a KV cache of their own; the last two share theirs.
CACHED_LAYERS = 4
_SPECIAL = ['<pad>', '<eos>', '<bos>', '<|turn>', '<turn|>', '<unk>']
_CHARACTERS = ['\n'] + [chr(code) for code in range(32, 127)]
# One id per token: a model's every id has its text.
VOCAB_SIZE = len(_SPECIAL) + len(_CHARACTERS)
_CHAT_TEMPLATE = (
    '{{ bos_token }}{% for message in messages %}<|turn>'
    "{{ 'model' if message.role == 'assistant' else message.role }}\n"
    '{{ message.content }}<turn|>\n{% endfor %}'
    '{% if add_generation_prompt %}<|turn>model\n{% endif %}'
)


def write(directory: pathlib.Path, seed: int = 0) -> pathlib.Path:
  """Writes the model directory; returns it.

  Args:
    directory: Where; created.
    seed: The seed of the random weights.

  Returns:
    The directory written.
  """
  directory.mkdir(parents=True, exist_ok=True)
  _tokenizer().save_pretrained(directory)
  config = transformers.Gemma4Config(
      text_config={
          'hidden_size': 64,
          'intermediate_size': 128,
          'num_hidden_layers': len(LAYER_TYPES),
          'num_attention_heads': 2,
          'num_key_value_heads': 1,
          'head_dim': 16,
          'global_head_dim': 32,
          'hidden_size_per_layer_input': 8,
          'vocab_size': VOCAB_SIZE,
          'vocab_size_per_layer_input': VOCAB_SIZE,
          'num_kv_shared_layers': len(LAYER_TYPES) - CACHED_LAYERS,
          'sliding_window': SLIDING_WINDOW,
          'layer_types': LAYER_TYPES,
          'max_position_embeddings': 512,
          'final_logit_softcapping': 30.0,
          'bos_token_id': BOS,
          'eos_token_id': EOS,
          'pad_token_id': PAD,
      },
      vision_config=None,
      audio_config=None,
      eos_token_id=[EOS, TURN_END],
  )
  torch.manual_seed(seed)
  model = transformers.Gemma4ForConditionalGeneration(config)
  model.save_pretrained(directory)
  transformers.GenerationConfig(
      bos_token_id=BOS,
      eos_token_id=[EOS, TURN_END],
      pad_token_id=PAD,
      do_sample=True,
      temperature=1.0,
      top_k=8,
      top_p=0.95,
  ).save_pretrained(directory)
  return directory


def _tokenizer() -> transformers.PreTrainedTokenizerFast:
  """Returns a tokenizer of one token per character."""
  vocabulary = {
      token: index for index, token in enumerate(_SPECIAL + _CHARACTERS)
  }
  backend = tokenizers.Tokenizer(
      models.WordLevel(vocab=vocabulary, unk_token='<unk>')
  )
  backend.pre_tokenizer = pre_tokenizers.Split(
      tokenizers.Regex('(?m:.)'), behavior='isolated'
  )
  backend.decoder = decoders.Fuse()
  backend.add_special_tokens(_SPECIAL)
  tokenizer = transformers.PreTrainedTokenizerFast(
      tokenizer_object=backend,
      bos_token='<bos>',
      eos_token='<eos>',
      pad_token='<pad>',
      unk_token='<unk>',
  )
  tokenizer.chat_template = _CHAT_TEMPLATE
  return tokenizer
