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

"""Synthetic capture fixtures: raw storage, exact tap joins, saved sessions."""

from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from ml_dtypes import bfloat16
from model_explorer_debugger.capture_importer import publish_capture
from model_explorer_debugger.fsutil import atomic_json
from model_explorer_debugger.runtime.litert_capture import collect
from model_explorer_debugger.session_registry import SessionRegistry
import numpy as np
from safetensors.numpy import save_file

TAPPED = "2" * 64


def tap(output_name="norm", signature="prefill", output=0):
  return dict(
      signature=signature,
      section_offset=100,
      subgraph=0,
      op=1,
      output=output,
      tensor=2 + output,
      output_name=output_name,
      tensor_name=(
          "model/layer_0/pre_attention_norm/output"
          if output == 0
          else "model/layer_0/other/output"
      ),
      shape=[1, 2],
      tensor_type=0,
  )


class SafetensorsCollectTests(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    self.capture = self.root / "capture"
    self.capture.mkdir()
    self.manifest = self.root / "manifest.json"
    self.destination = self.root / "export"
    self.array = np.array([[1, 2]], dtype=np.float32)
    self.set_taps([tap()])

  def set_taps(self, taps):
    atomic_json(
        self.manifest,
        {
            "format_version": 1,
            "source_sha256": "1" * 64,
            "tapped_sha256": TAPPED,
            "taps": taps,
        },
    )

  def shard(self, name, tensors=None, metadata=None):
    path = self.capture / name
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors if tensors is not None else {"post_norm": self.array},
        path,
        metadata=metadata
        if metadata is not None
        else {"signature": "prefill", "step": "15"},
    )
    return path

  def collect(self):
    return collect(self.capture, self.manifest, self.destination, "CPU")

  def test_raw_shard_is_preserved_and_multiple_selected_keys_share_one_path(
      self,
  ):
    self.set_taps([tap(), tap("second", output=1), tap(signature="decode")])
    source = self.shard(
        "session-a/outputs.safetensors",
        {
            "post_norm": self.array,
            "post_second": self.array + 3,
            "unselected": np.array([7], dtype=np.int32),
        },
    )
    raw = source.read_bytes()
    result = self.collect()
    self.assertEqual(result["format_version"], 2)
    self.assertEqual(result["tapped_sha256"], TAPPED)
    self.assertEqual(result["backend_requested"], "CPU")
    self.assertEqual(result["uncaptured_signatures"], ["decode"])
    self.assertEqual(len(result["tensors"]), 2)
    self.assertEqual(
        {r["key"] for r in result["tensors"]}, {"post_norm", "post_second"}
    )
    paths = {r["path"] for r in result["tensors"]}
    self.assertEqual(len(paths), 1)
    for record in result["tensors"]:
      self.assertEqual(record["format"], "safetensors")
      self.assertEqual(record["signature"], "prefill")
      self.assertEqual(record["step"], 15)
      self.assertEqual(record["phase"], "prefill")
      self.assertEqual(record["session"], "session-a")
      self.assertEqual(record["shape"], [1, 2])
      self.assertEqual(record["dtype"], "float32")
      self.assertEqual(Path(record["source"]), source.resolve())
      self.assertFalse(Path(record["path"]).is_absolute())
      self.assertEqual(Path(record["path"]).suffix, ".safetensors")
      self.assertEqual((self.destination / record["path"]).read_bytes(), raw)
    self.assertEqual(len(list(self.destination.rglob("*.safetensors"))), 1)
    self.assertEqual(list(self.destination.rglob("*.npy")), [])
    self.assertEqual(
        json.loads((self.destination / "capture_index.json").read_text()),
        result,
    )

  def test_signature_and_key_must_match_exactly(self):
    self.shard(
        "wrong-signature.safetensors",
        metadata={"signature": "prefill_similar", "step": "15"},
    )
    self.shard(
        "wrong-key.safetensors",
        {"post_norm_extra": self.array, "norm": self.array},
    )
    with self.assertRaises(ValueError):
      self.collect()
    self.assertFalse(self.destination.exists())

  def test_prefill_and_decode_keep_runtime_steps_and_distinct_shards(self):
    self.set_taps([tap(), tap(signature="decode")])
    self.shard("session-a/prefill.safetensors")
    self.shard(
        "session-a/decode.safetensors",
        metadata={"signature": "decode", "step": "22"},
    )
    records = self.collect()["tensors"]
    self.assertEqual(
        {(r["signature"], r["phase"], r["step"]) for r in records},
        {("prefill", "prefill", 15), ("decode", "decode", 22)},
    )
    self.assertEqual(len({r["path"] for r in records}), 2)

  def test_incomplete_selected_outputs_cannot_be_exported(self):
    self.set_taps([tap(), tap("second", output=1)])
    self.shard("session-a/outputs.safetensors")
    with self.assertRaisesRegex(ValueError, "Incomplete"):
      self.collect()
    self.assertFalse(self.destination.exists())

  def test_duplicate_identity_in_two_shards_cannot_be_exported(self):
    self.shard("session-a/first.safetensors")
    self.shard("session-a/duplicate.safetensors")
    with self.assertRaisesRegex(ValueError, "Duplicate"):
      self.collect()
    self.assertFalse(self.destination.exists())

  def test_equal_steps_in_distinct_runtime_sessions_are_not_duplicates(self):
    self.shard("session-a/outputs.safetensors")
    self.shard("session-b/outputs.safetensors")
    records = self.collect()["tensors"]
    self.assertEqual(
        {r["session"] for r in records}, {"session-a", "session-b"}
    )
    self.assertEqual(len({r["path"] for r in records}), 2)

  def test_missing_negative_and_malformed_steps_are_rejected(self):
    for step in (None, "-1", "invalid"):
      with self.subTest(step=step):
        metadata = {"signature": "prefill"}
        if step is not None:
          metadata["step"] = step
        self.shard("outputs.safetensors", metadata=metadata)
        with self.assertRaises(ValueError):
          self.collect()
        self.assertFalse(self.destination.exists())

  def test_shape_and_dtype_must_match_the_manifest(self):
    for array in (
        np.array([1, 2], dtype=np.float32),
        np.array([[1, 2]], dtype=np.float16),
    ):
      with self.subTest(shape=array.shape, dtype=str(array.dtype)):
        self.shard("outputs.safetensors", {"post_norm": array})
        with self.assertRaises(ValueError):
          self.collect()
        self.assertFalse(self.destination.exists())

  def test_bfloat16_capture_retains_dtype_and_raw_special_value_bits(self):
    self.set_taps([{**tap(), "tensor_type": 18, "shape": [1, 4]}])
    array = np.array([[0, 0x8000, 0x7FC1, 0xFFC2]], dtype=np.uint16).view(
        bfloat16
    )
    source = self.shard("outputs.safetensors", {"post_norm": array})
    raw = source.read_bytes()
    record = self.collect()["tensors"][0]
    self.assertEqual(record["dtype"], "bfloat16")
    self.assertEqual(record["shape"], [1, 4])
    self.assertEqual((self.destination / record["path"]).read_bytes(), raw)

  def test_existing_export_is_not_overwritten(self):
    self.destination.mkdir()
    marker = self.destination / "existing.txt"
    marker.write_text("keep")
    self.shard("outputs.safetensors")
    with self.assertRaises(ValueError):
      self.collect()
    self.assertEqual(marker.read_text(), "keep")

  def test_reference_export_points_to_raw_shards_without_copying_payload(self):
    self.set_taps([tap(), tap("second", output=1)])
    source = self.shard(
        "session-a/outputs.safetensors",
        {
            "post_norm": self.array,
            "post_second": self.array + 3,
        },
    )
    raw = source.read_bytes()
    result = collect(
        self.capture, self.manifest, self.destination, "CPU", reference=True
    )
    self.assertEqual(result["format_version"], 2)
    self.assertEqual(result["tensor_root"], "run")
    self.assertEqual(len(result["tensors"]), 2)
    for record in result["tensors"]:
      self.assertEqual(record["format"], "safetensors")
      self.assertEqual(
          (self.destination.parent / record["path"]).resolve(), source.resolve()
      )
    self.assertEqual(source.read_bytes(), raw)
    self.assertEqual(list(self.destination.rglob("*.safetensors")), [])
    self.assertEqual(list(self.destination.rglob("*.npy")), [])

  def test_reference_export_rejects_capture_outside_its_run(self):
    self.shard("session-a/outputs.safetensors")
    with self.assertRaises(ValueError):
      collect(
          self.capture,
          self.manifest,
          self.root / "different-run" / "export",
          "CPU",
          reference=True,
      )
    self.assertFalse((self.root / "different-run" / "export").exists())


class SafetensorsPublishTests(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    self.registry = SessionRegistry(self.root / "workspace")
    model = self.root / "model.litertlm"
    model.write_bytes(b"synthetic model fixture; no inference was run")
    semantic = self.root / "semantic.json"
    atomic_json(
        semantic,
        {
            "semantic_graph": [{
                "kind": "decoder",
                "inputs": [],
                "nodes": [{
                    "id": "norm.attn.in",
                    "label": "Norm",
                    "namespace": "",
                    "incomingEdges": [],
                }],
                "anchors": [],
            }],
            "layers": [{"def": 0, "attrs": {}}],
        },
    )
    artifact = self.registry.register_model(model, semantic=semantic)
    self.record = self.registry.manage(
        "create",
        {
            "name": "Synthetic raw capture",
            "model": "Synthetic model",
            "runs": [
                dict(
                    id=run,
                    runtime="LiteRT-LM",
                    backend=backend,
                    artifact=artifact,
                    source="registered",
                )
                for run, backend in (("ref", "CPU"), ("target", "CPU"))
            ],
        },
    )
    self.artifacts = {
        run["id"]: self.registry.resolve_run(run) for run in self.record["runs"]
    }

  def fixture(self, identity="synthetic-job-1", delta=1, reference=False):
    job = (
        self.registry.root / "sessions" / self.record["id"] / "jobs" / identity
    )
    results, sources = {}, {}
    for run in ("ref", "target"):
      directory = job / run
      raw = directory / "capture" / "runtime-session"
      raw.mkdir(parents=True)
      manifest = directory / "manifest.json"
      atomic_json(
          manifest,
          {
              "format_version": 1,
              "source_sha256": "1" * 64,
              "tapped_sha256": TAPPED,
              "taps": [tap(), tap("second", output=1)],
          },
      )
      source = raw / "outputs.safetensors"
      save_file(
          {
              "post_norm": np.array(
                  [[1, 2 + (delta if run == "target" else 0)]], dtype=np.float32
              ),
              "post_second": np.array([[3, 4]], dtype=np.float32),
          },
          source,
          metadata={"signature": "prefill", "step": "15"},
      )
      collect(
          directory / "capture",
          manifest,
          directory / "export",
          "CPU",
          reference=reference,
      )
      results[run] = dict(
          model_sha256="same-synthetic-model",
          input="hello",
          output="synthetic output",
          messages=[],
      )
      sources[run] = source
    return job, results, sources

  def publish(self, job, results):
    capture = publish_capture(
        self.registry, self.record, job, results, self.artifacts
    )
    self.record = self.registry.update(
        self.record["id"], capture=capture, has_capture=True
    )
    return self.registry.capture_store(self.record["id"])

  def test_publish_compares_raw_tensors_and_reloads_both_turns(self):
    job, results, sources = self.fixture()
    store = self.publish(job, results)
    first_root = store.root
    first_bytes = {
        r["path"]: (store.root / r["path"]).read_bytes() for r in store.tensors
    }
    self.assertEqual(len(store.tensors), 4)
    self.assertEqual(len(first_bytes), 2)
    for tensor in store.tensors:
      self.assertEqual(tensor["format"], "safetensors")
      self.assertEqual(
          (store.root / tensor["path"]).read_bytes(),
          sources[tensor["run"]].read_bytes(),
      )
      self.assertIn(tensor["key"], ("post_norm", "post_second"))
      np.testing.assert_array_equal(
          store.load(tensor),
          [[3, 4]]
          if tensor["key"] == "post_second"
          else [[1, 3 if tensor["run"] == "target" else 2]],
      )
    row = store.compare_batch(0)["rows"][0]
    self.assertEqual(row["status"], "ok")
    self.assertEqual(row["metrics"]["Max abs error"]["value"], 1.0)
    job, results, _ = self.fixture("synthetic-job-2", delta=2)
    with patch(
        "model_explorer_debugger.capture_importer.shutil.copy2",
        wraps=shutil.copy2,
    ) as copy:
      store = self.publish(job, results)
    copied_destinations = Counter(
        str(call.args[1]) for call in copy.call_args_list
    )
    self.assertTrue(copied_destinations)
    self.assertTrue(all(count == 1 for count in copied_destinations.values()))
    self.assertEqual(len(store.tensors), 8)
    self.assertEqual(len({r["path"] for r in store.tensors}), 4)
    self.assertEqual([turn["n"] for turn in store.session["turns"]], [1, 2])
    self.assertEqual(list(store.root.rglob("*.npy")), [])
    for path, raw in first_bytes.items():
      self.assertEqual((first_root / path).read_bytes(), raw)
      self.assertEqual((store.root / path).read_bytes(), raw)
    reopened = SessionRegistry(self.registry.root).capture_store(
        self.record["id"]
    )
    for batch, error in ((0, 1.0), (1, 2.0)):
      row = reopened.compare_batch(batch)["rows"][0]
      self.assertEqual(row["status"], "ok")
      self.assertEqual(row["metrics"]["Max abs error"]["value"], error)

  def test_legacy_npy_export_is_rejected(self):
    job, results, _ = self.fixture()
    index_path = job / "ref" / "export" / "capture_index.json"
    index = json.loads(index_path.read_text())
    index.pop("format_version")
    for i, tensor in enumerate(index["tensors"]):
      tensor.pop("format")
      tensor.pop("key")
      tensor["path"] = f"tensors/legacy-{i}.npy"
      array = np.array(
          [[1, 2]] if tensor["output"] == 0 else [[3, 4]], dtype=np.float32
      )
      np.save(index_path.parent / tensor["path"], array, allow_pickle=False)
    atomic_json(index_path, index)
    with self.assertRaises(ValueError):
      publish_capture(self.registry, self.record, job, results, self.artifacts)
    self.assertFalse((job.parent.parent / "captures" / job.name).exists())
    self.assertFalse(self.registry.get(self.record["id"])["has_capture"])

  def test_metadata_only_run_export_publishes_self_contained_raw_capture(self):
    job, results, sources = self.fixture(reference=True)
    original = {run: path.read_bytes() for run, path in sources.items()}
    store = self.publish(job, results)
    self.assertEqual(len({t["path"] for t in store.tensors}), 2)
    for run in ("ref", "target"):
      self.assertEqual(list((job / run / "export").rglob("*.safetensors")), [])
    shutil.rmtree(job)
    reopened = SessionRegistry(self.registry.root).capture_store(
        self.record["id"]
    )
    self.assertEqual(reopened.compare_batch(0)["rows"][0]["status"], "ok")
    for tensor in reopened.tensors:
      self.assertEqual(
          (reopened.root / tensor["path"]).read_bytes(), original[tensor["run"]]
      )

  def test_out_of_root_and_symlink_export_paths_cannot_publish(self):
    for kind in ("relative-traversal", "absolute", "symlink"):
      with self.subTest(kind=kind):
        job, results, sources = self.fixture("invalid-" + kind)
        outside = job / "outside.safetensors"
        outside.write_bytes(sources["ref"].read_bytes())
        index_path = job / "ref" / "export" / "capture_index.json"
        if kind == "relative-traversal":
          invalid = "../../outside.safetensors"
        elif kind == "absolute":
          invalid = str(outside)
        else:
          (index_path.parent / "escape.safetensors").symlink_to(outside)
          invalid = "escape.safetensors"
        index = json.loads(index_path.read_text())
        index["tensors"][0]["path"] = invalid
        atomic_json(index_path, index)
        with self.assertRaises(ValueError):
          publish_capture(
              self.registry, self.record, job, results, self.artifacts
          )
        destination = job.parent.parent / "captures" / job.name
        self.assertFalse(destination.exists())
        self.assertFalse(self.registry.get(self.record["id"])["has_capture"])

  def test_missing_tensor_key_and_false_metadata_cannot_publish(self):
    for i, change in enumerate(
        ({"key": "absent"}, {"key": ""}, {"shape": [2]}, {"dtype": "float16"})
    ):
      with self.subTest(change=change):
        job, results, _ = self.fixture(f"malformed-{i}")
        index_path = job / "ref" / "export" / "capture_index.json"
        index = json.loads(index_path.read_text())
        index["tensors"][0].update(change)
        atomic_json(index_path, index)
        with self.assertRaises((ValueError, KeyError)):
          publish_capture(
              self.registry, self.record, job, results, self.artifacts
          )
        self.assertFalse((job.parent.parent / "captures" / job.name).exists())


if __name__ == "__main__":
  unittest.main()
