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

"""Synthetic Runner folders exercise the real CLI importer without inference."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from model_explorer_debugger.fsutil import atomic_json
from model_explorer_debugger.tensor_io import load_tensor
import numpy as np
from safetensors.numpy import save_file

CLI_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/server/package/model_explorer_debugger/runtime/import_capture.py"
)
SPEC = importlib.util.spec_from_file_location(
    "synthetic_ios_import_capture", CLI_PATH
)
IMPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORTER)


class SafetensorsIOSImportTests(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    self.source = self.root / "runner-export"
    (self.source / "tensors").mkdir(parents=True)
    self.model = self.root / "prepared.litertlm"
    self.model.write_bytes(
        b"Synthetic model; verify_taps is mocked, no native inference"
    )
    checksum = hashlib.sha256(self.model.read_bytes()).hexdigest()
    job_id = "11111111-2222-4333-8444-555555555555"
    taps, tensors = [], []
    self.payloads, self.arrays = {}, {}
    for i, signature in enumerate(("prefill", "decode")):
      taps.append(
          dict(
              signature=signature,
              output_name="norm",
              section_offset=100,
              subgraph=i,
              op=1,
              output=0,
              tensor=2,
              tensor_name="model/layer_0/pre_attention_norm/output",
              shape=[1, 2],
              tensor_type=0,
          )
      )
      path = f"tensors/{signature}.safetensors"
      array = np.array([[1 + i, 2 + i]], dtype=np.float32)
      save_file(
          {"post_norm": array},
          self.source / path,
          metadata={"signature": signature, "step": str(15 + i)},
      )
      raw = (self.source / path).read_bytes()
      tensors.append(
          dict(
              path=path,
              sha256=hashlib.sha256(raw).hexdigest(),
              signature=signature,
              step=15 + i,
              key="post_norm",
              shape=[1, 2],
              dtype="F32",
          )
      )
      self.payloads[signature], self.arrays[signature] = raw, array
    self.manifest = dict(
        format_version=1,
        source_sha256="b" * 64,
        tapped_sha256="a" * 64,
        taps=taps,
    )
    self.job = dict(
        formatVersion=1,
        id=job_id,
        modelSHA256=checksum,
        prompt="hello",
        contextLength=1024,
        maxOutputTokens=2,
        manifest=self.manifest,
    )
    self.result = dict(
        formatVersion=1,
        jobID=job_id,
        modelSHA256=checksum,
        platform="synthetic-test",
        operatingSystem="synthetic",
        hardwareIdentifier="synthetic",
        backend="CPU",
        debuggerEnabled=True,
        sampler=dict(topK=1, topP=1, temperature=0, seed=0),
        thinkingEnabled=False,
        speculativeDecodingEnabled=False,
        input="hello",
        output="synthetic output",
        maxOutputTokens=2,
        contextLength=1024,
        elapsedSecondsWithCapture=0,
        tensors=tensors,
        uncapturedPoints=[],
    )
    atomic_json(self.source / "job.json", self.job)
    atomic_json(self.source / "result.json", self.result)
    atomic_json(self.source / "runtime-build.json", {"synthetic": True})

  def invoke(self, output):
    argv = [
        str(CLI_PATH),
        "--runtime-root",
        str(self.root / "unused-runtime"),
        "--capture",
        str(self.source),
        "--model",
        str(self.model),
        "--output",
        str(output),
    ]
    original_path = sys.path[:]
    stdout = io.StringIO()
    try:
      with (
          patch.object(sys, "argv", argv),
          redirect_stdout(stdout),
          patch(
              "model_explorer_debugger.runtime.litert_lm_adapter.verify_taps",
              return_value=deepcopy(self.manifest),
          ) as verify,
      ):
        IMPORTER.main()
        self.assertEqual(verify.call_count, 1)
        self.assertEqual(verify.call_args.args[0], self.model)
    finally:
      sys.path[:] = original_path
    return stdout.getvalue()

  def test_cli_export_keeps_raw_bytes_and_all_paths_survive_staging_cleanup(
      self,
  ):
    output = self.root / "saved-export"
    message = self.invoke(output)
    self.assertIn("Validated 2 tensors", message)
    index = json.loads((output / "capture_index.json").read_text())
    self.assertEqual(index["format_version"], 2)
    self.assertEqual(index["tensor_root"], "export")
    self.assertEqual(len(index["tensors"]), 2)
    self.assertEqual(len(list(output.rglob("*.safetensors"))), 2)
    self.assertEqual(list(output.rglob("*.npy")), [])
    self.assertEqual(list(self.root.glob(".ios-import-*")), [])
    self.assertEqual(json.loads((output / "job.json").read_text()), self.job)
    self.assertEqual(
        json.loads((output / "runner-result.json").read_text()), self.result
    )
    self.assertEqual(
        json.loads((output / "runtime-build.json").read_text()),
        {"synthetic": True},
    )
    for tensor in index["tensors"]:
      self.assertEqual(tensor["format"], "safetensors")
      self.assertEqual(tensor["key"], "post_norm")
      self.assertEqual(tensor["dtype"], "float32")
      self.assertEqual(tensor["path"], f"raw/{tensor['signature']}.safetensors")
      self.assertEqual(
          Path(tensor["source"]), (output / tensor["path"]).resolve()
      )
      self.assertNotIn(".ios-import-", tensor["source"])
      self.assertEqual(
          (output / tensor["path"]).read_bytes(),
          self.payloads[tensor["signature"]],
      )
      self.assertEqual(
          (
              self.source / f"tensors/{tensor['signature']}.safetensors"
          ).read_bytes(),
          self.payloads[tensor["signature"]],
      )
    # The saved export has no dependency on the phone folder or temporary
    # staging.
    shutil.rmtree(self.source)
    for tensor in index["tensors"]:
      np.testing.assert_array_equal(
          load_tensor(output, tensor), self.arrays[tensor["signature"]]
      )

  def test_declared_tensor_metadata_must_agree_with_actual_raw_payload(self):
    for i, change in enumerate((
        {"signature": "wrong"},
        {"step": 99},
        {"key": "wrong"},
        {"shape": [2]},
        {"dtype": "F16"},
    )):
      with self.subTest(change=change):
        result = deepcopy(self.result)
        result["tensors"][0].update(change)
        atomic_json(self.source / "result.json", result)
        output = self.root / f"invalid-metadata-{i}"
        with self.assertRaisesRegex(ValueError, "metadata|dtype"):
          self.invoke(output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".ios-import-*")), [])

  def test_changed_payload_is_rejected_by_checksum_before_publication(self):
    source = self.source / self.result["tensors"][0]["path"]
    source.write_bytes(source.read_bytes() + b"changed")
    output = self.root / "invalid-checksum"
    with self.assertRaisesRegex(ValueError, "checksum"):
      self.invoke(output)
    self.assertFalse(output.exists())
    self.assertEqual(list(self.root.glob(".ios-import-*")), [])


if __name__ == "__main__":
  unittest.main()
