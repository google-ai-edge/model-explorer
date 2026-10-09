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

"""Artifact writers for manifests, safetensors shards, and capture indices."""

import json
import pathlib
from typing import Any

import safetensors.torch
import torch


def append_manifest_rows(
    target_dir: pathlib.Path, manifest_rows: list[dict[str, Any]]
) -> None:
  """Appends JSON rows to target_dir/manifest.jsonl.

  Args:
    target_dir: Directory where manifest.jsonl is located.
    manifest_rows: List of metadata dictionaries to serialize as JSON lines.
  """
  if not manifest_rows:
    return
  target_dir.mkdir(parents=True, exist_ok=True)
  manifest_path = target_dir / "manifest.jsonl"
  with manifest_path.open("a", encoding="utf-8") as f:
    for row in manifest_rows:
      f.write(json.dumps(row) + "\n")


def write_safetensors_shard(
    target_dir: pathlib.Path, shard_name: str, tensors: dict[str, torch.Tensor]
) -> None:
  """Saves a tensor mapping to target_dir/shard_name.

  Args:
    target_dir: Directory where the safetensors file is stored.
    shard_name: Name of the safetensors shard file.
    tensors: Mapping of tensor names to cloned CPU tensors.
  """
  if not tensors:
    return
  target_dir.mkdir(parents=True, exist_ok=True)
  safetensors.torch.save_file(tensors, target_dir / shard_name)


def write_module_shard(
    out_dir: pathlib.Path,
    shard_name: str,
    tensors: dict[str, torch.Tensor],
    manifest_rows: list[dict[str, Any]],
) -> None:
  """Saves module tensors and appends rows to manifest.jsonl.

  Args:
    out_dir: Base directory for the active capture run.
    shard_name: Name of the safetensors file (e.g. 'shard-00000.safetensors').
    tensors: Mapping of tensor slot names to cloned CPU tensors.
    manifest_rows: Manifest metadata rows describing each module tensor.
  """
  write_safetensors_shard(out_dir, shard_name, tensors)
  append_manifest_rows(out_dir, manifest_rows)


def append_boundary_manifest(
    out_dir: pathlib.Path, manifest_rows: list[dict[str, Any]]
) -> None:
  """Appends boundary manifest rows to boundaries/manifest.jsonl.

  Args:
    out_dir: Base directory for the active capture run.
    manifest_rows: Manifest metadata rows describing boundary tensors.
  """
  append_manifest_rows(out_dir / "boundaries", manifest_rows)


def write_boundary_shard(
    out_dir: pathlib.Path,
    shard_name: str,
    tensors: dict[str, torch.Tensor],
    manifest_rows: list[dict[str, Any]] | None = None,
) -> None:
  """Saves boundary tensors and optionally appends manifest rows.

  Args:
    out_dir: Base directory for the active capture run.
    shard_name: Name of the safetensors file (e.g. 'boundaries.safetensors').
    tensors: Mapping of boundary slot names to cloned CPU tensors.
    manifest_rows: Optional metadata rows describing boundary tensors.
  """
  write_safetensors_shard(out_dir / "boundaries", shard_name, tensors)
  if manifest_rows:
    append_manifest_rows(out_dir / "boundaries", manifest_rows)


def write_kv_shard(
    out_dir: pathlib.Path,
    shard_name: str,
    tensors: dict[str, torch.Tensor],
    manifest_rows: list[dict[str, Any]],
) -> None:
  """Saves KV cache tensors and appends rows to kv/manifest.jsonl.

  Args:
    out_dir: Base directory for the active capture run.
    shard_name: Name of the safetensors file (e.g. 'kv_00000.safetensors').
    tensors: Mapping of KV cache slot names to cloned CPU tensors.
    manifest_rows: Manifest metadata rows describing each key/value tensor.
  """
  write_safetensors_shard(out_dir / "kv", shard_name, tensors)
  append_manifest_rows(out_dir / "kv", manifest_rows)


def write_forward_index(
    out_dir: pathlib.Path, forwards: list[dict[str, Any]]
) -> None:
  """Writes forward_index.json.

  Args:
    out_dir: Base directory for the active capture run.
    forwards: List of forward pass summary metadata records.
  """
  out_dir.mkdir(parents=True, exist_ok=True)
  index_path = out_dir / "forward_index.json"
  index_path.write_text(
      json.dumps({"forwards": forwards}, indent=2) + "\n", encoding="utf-8"
  )


def write_tokens(
    out_dir: pathlib.Path, token_records: list[dict[str, Any]]
) -> None:
  """Writes tokens.jsonl.

  Args:
    out_dir: Base directory for the active capture run.
    token_records: Sequential token generation and consumption records.
  """
  out_dir.mkdir(parents=True, exist_ok=True)
  tokens_path = out_dir / "tokens.jsonl"
  content = "".join(json.dumps(r) + "\n" for r in token_records)
  tokens_path.write_text(content, encoding="utf-8")


def write_generation(out_dir: pathlib.Path, generation: dict[str, Any]) -> None:
  """Writes generation.json.

  Args:
    out_dir: Base directory for the active capture run.
    generation: Summary dictionary containing generation status and token
      counts.
  """
  out_dir.mkdir(parents=True, exist_ok=True)
  gen_path = out_dir / "generation.json"
  gen_path.write_text(json.dumps(generation, indent=2) + "\n", encoding="utf-8")


def write_kv_index(
    out_dir: pathlib.Path, snapshots: list[dict[str, Any]]
) -> None:
  """Writes kv_index.json.

  Args:
    out_dir: Base directory for the active capture run.
    snapshots: List of KV cache snapshot metadata dictionaries.
  """
  out_dir.mkdir(parents=True, exist_ok=True)
  kv_path = out_dir / "kv_index.json"
  kv_path.write_text(
      json.dumps({"snapshots": snapshots}, indent=2) + "\n", encoding="utf-8"
  )
