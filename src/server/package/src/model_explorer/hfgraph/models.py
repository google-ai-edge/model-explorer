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

"""Model loading: text decoder only, optional shrunken config.

Provides utilities to load text decoder models and optional shrunken configs
for structure checks.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from typing import Any

import torch
from transformers import AutoConfig
from transformers import AutoModelForCausalLM

FAMILIES = {
    "qwen3_5": (
        "transformers.models.qwen3_5.modeling_qwen3_5",
        "Qwen3_5ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
    ),
    "qwen3_5_text": (
        "transformers.models.qwen3_5.modeling_qwen3_5",
        "Qwen3_5ForCausalLM",
        None,
    ),
    "qwen4_exp": (
        "transformers.models.qwen4_exp.modeling_qwen4_exp",
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForConditionalGeneration",
    ),
    "qwen4_exp_text": (
        "transformers.models.qwen4_exp.modeling_qwen4_exp",
        "Qwen4ExpForCausalLM",
        None,
    ),
}


def text_config(hf_id: str) -> tuple[Any, Any]:
  """Loads the top-level and text decoder configurations for a model.

  Args:
    hf_id: HuggingFace model identifier.

  Returns:
    A tuple of (full_config, text_config).
  """
  # trust_remote_code=False here and below: never run code shipped with a
  # checkpoint. Later stages also evaluate branch conditions parsed from the
  # model source (prune.evaluate), which is only safe while that source is the
  # installed transformers package.
  cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=False)
  return cfg, getattr(cfg, "text_config", cfg)


def config_hash(tcfg: Any) -> str:
  """Computes a SHA-1 hash prefix of the serialized configuration."""
  return hashlib.sha1(
      json.dumps(tcfg.to_dict(), sort_keys=True, default=str).encode()
  ).hexdigest()[:12]


def shrink(tcfg: Any) -> Any:
  """Shrinks layer and attention counts for fast structural tracing."""
  n = 4
  tcfg.num_hidden_layers = n
  if getattr(tcfg, "layer_types", None):
    tcfg.layer_types = tcfg.layer_types[:n]
  tcfg.hidden_size = 256
  tcfg.intermediate_size = 512
  tcfg.num_attention_heads = 4
  tcfg.num_key_value_heads = 2
  tcfg.head_dim = 64
  for k, v in dict(
      linear_num_value_heads=4,
      linear_num_key_heads=4,
      linear_key_head_dim=32,
      linear_value_head_dim=32,
  ).items():
    if hasattr(tcfg, k):
      setattr(tcfg, k, v)
  tcfg.vocab_size = 1024
  for k, v in dict(
      num_experts=8,
      num_experts_per_tok=2,
      moe_intermediate_size=64,
      shared_expert_intermediate_size=64,
      ngram_vocab_size_base=512,
      ple_embed_dim=32,
      indexer_head_dim=32,
      indexer_n_heads=2,
      indexer_kv_heads=1,
      indexer_budget=8,
      mtp_num_hidden_layers=0,
  ).items():
    if hasattr(tcfg, k):
      setattr(tcfg, k, v)
  return tcfg


def _import(path: str, name: str) -> Any:
  """Imports an attribute dynamically by module path and attribute name."""
  return getattr(importlib.import_module(path), name)


def _load_pretrained(model_cls: Any, hf_id: str) -> Any:
  """Loads a pretrained model in float32 with remote code disabled."""
  return model_cls.from_pretrained(
      hf_id,
      dtype=torch.float32,
      torch_dtype=torch.float32,
      trust_remote_code=False,
  )


def build_text_model(
    hf_id: str, tiny: bool, device: str = "cpu"
) -> tuple[torch.nn.Module, Any]:
  """Instantiates the text decoder causal LM model.

  Args:
    hf_id: HuggingFace model identifier.
    tiny: If True, uses random weights with a shrunken config.
    device: Device to place the model on ('cpu' or 'meta').

  Returns:
    A tuple of (model, text_config).
  """
  cfg, tcfg = text_config(hf_id)
  fam = FAMILIES.get(tcfg.model_type) or FAMILIES.get(cfg.model_type)
  if tiny:
    tcfg = shrink(tcfg)
  if fam is None:
    if device == "meta" or tiny:
      with torch.device(device if device == "meta" else "cpu"):
        torch.manual_seed(0)
        return (
            AutoModelForCausalLM.from_config(
                tcfg, trust_remote_code=False
            ).eval(),
            tcfg,
        )
    return _load_pretrained(AutoModelForCausalLM, hf_id).eval(), tcfg
  mod_path, lm_cls, full_cls = fam
  model_cls = _import(mod_path, lm_cls)
  if device == "meta" or tiny:
    with torch.device(device if device == "meta" else "cpu"):
      torch.manual_seed(0)
      return model_cls(tcfg).eval().float(), tcfg
  if full_cls:
    full = _load_pretrained(_import(mod_path, full_cls), hf_id)
    lm = model_cls(tcfg)
    lm.model.load_state_dict(full.model.language_model.state_dict())
    lm.lm_head.load_state_dict(full.lm_head.state_dict())
    del full
    return lm.eval(), tcfg
  lm = _load_pretrained(model_cls, hf_id)
  return lm.eval(), tcfg
