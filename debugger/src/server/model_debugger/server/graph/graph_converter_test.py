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

"""Tests of the real converter, in its child, and of what it writes.

The child's own life (stopped, timed out, its parent killed, the converter
missing) is tested with a fake converter package put first on the path the
child imports from: it sleeps, or ends itself, as each test needs.
"""

import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import textwrap
import time
from unittest import mock

from absl.testing import absltest
from model_debugger.server.graph import architecture
from model_debugger.server.graph import graph_converter
from model_debugger.server.graph import model_graph
from model_debugger.server.tap import tap_scan
from model_debugger.server.testing import tap_fixtures
from model_debugger.server.testing import test_main

# tap_scan's rule before the layers came from the graph: the first
# `layer_<N>/` of a tensor's first name.
_OLD_LAYER = re.compile(r'(?:^|/)layer_(\d+)/')
_TIMEOUT = 600.0
# A fake converter that writes its process id, then sleeps.
_SLEEPS = """
import os, pathlib, time
class VisualizeConfig:
    pass
def ConvertFlatbufferDirectlyToJson(config, path):
    pathlib.Path({pid!r}).write_text(str(os.getpid()))
    time.sleep(60)
"""
# One that ends itself by a signal.
_SIGNALS = """
import os, signal
class VisualizeConfig:
    pass
def ConvertFlatbufferDirectlyToJson(config, path):
    os.kill(os.getpid(), signal.SIGTERM)
"""
# One that is not there: importing it fails as a missing wheel does.
_MISSING = """
raise ImportError("No module named 'ai_edge_model_explorer_adapter._ffi'")
"""
# The server's process, as a child of the test, converting with the fake.
_PARENT = """
import pathlib, sys
from model_debugger.server.graph import graph_converter
graph_converter.convert(
    pathlib.Path(sys.argv[1]), {'decode': 0}, pathlib.Path(sys.argv[2]),
    lambda: False, 600.0)
"""


def _subgraphs(data: bytes) -> dict[str, int]:
  signatures, _, _ = tap_scan.read_model(data)
  return {s.name: s.subgraph for s in signatures}


def _gone(pid: int, seconds: float) -> bool:
  """Whether a process is gone, or goes within `seconds`."""
  deadline = time.monotonic() + seconds
  while True:
    try:
      os.kill(pid, 0)
    except ProcessLookupError:
      return True
    if time.monotonic() > deadline:
      return False
    time.sleep(0.05)


def _kill(pid: int) -> None:
  try:
    os.kill(pid, signal.SIGKILL)
  except ProcessLookupError:
    pass


class GraphConverterTest(absltest.TestCase):
  """Tests of graph_converter.convert(), on the fixture and a real model."""

  def setUp(self):
    """The fixture model, a .litertlm file."""
    super().setUp()
    self.root = pathlib.Path(self.create_tempdir().full_path)
    self.model = self.root / 'model.litertlm'
    tap_fixtures.write_model(self.model)
    self.pid = self.root / 'converter.pid'

  def _fake(self, body: str) -> pathlib.Path:
    """Puts a fake converter first on the path the child imports from."""
    root = self.root / 'fake'
    package = root / 'ai_edge_model_explorer_adapter'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text(
        textwrap.dedent(body.format(pid=str(self.pid)))
    )
    self.enter_context(mock.patch.object(sys, 'path', [str(root), *sys.path]))
    return root

  def _converter_pid(self) -> int:
    """The fake converter's process id, once it wrote it."""
    deadline = time.monotonic() + 60
    while not self.pid.is_file() or not self.pid.read_text():
      self.assertLess(time.monotonic(), deadline, 'no converter ran')
      time.sleep(0.05)
    pid = int(self.pid.read_text())
    self.addCleanup(_kill, pid)
    return pid

  def test_converts_each_signature_in_a_child(self):
    """Node i of a signature is its op i; its outputs are its tensors."""
    out = self.root / 'graphs.json'
    subgraphs = _subgraphs(tap_fixtures.text_model())
    graph_converter.convert(self.model, subgraphs, out, lambda: False, _TIMEOUT)
    graphs = json.loads(out.read_text())
    self.assertEqual(list(graphs), list(tap_fixtures.SIGNATURES))
    stems = model_graph.layer_stems(graphs).stems
    self.assertEqual(set(stems.values()), {'layer'})
    _, points, _ = tap_scan.read_model(tap_fixtures.text_model(), stems)
    for name, graph in graphs.items():
      annotated = model_graph.annotate(graph, points, name)
      ids = set()
      for node in model_graph.operators(annotated):
        for entry in node['outputsMetadata']:
          ids.add(model_graph.attrs(entry).get('point_id'))
      self.assertEqual(ids, {p.id for p in points if p.signature == name})
    # The converter's logging goes to a file of its own; nothing else
    # is left beside the graphs, the structure copies it read deleted.
    self.assertEqual(
        graph_converter.log_path(out), self.root / 'graphs.json.log'
    )
    self.assertTrue(graph_converter.log_path(out).is_file())
    self.assertEqual(
        sorted(p.name for p in self.root.iterdir()),
        ['graphs.json', 'graphs.json.log', 'model.litertlm'],
    )

  def test_a_stopped_conversion_ends_its_child(self):
    """stopped() kills the child, and waits for it; nothing is written."""
    self._fake(_SLEEPS)
    out = self.root / 'graphs.json'
    with self.assertRaises(graph_converter.StoppedError):
      graph_converter.convert(
          self.model,
          {'decode': 0},
          out,
          lambda: self.pid.is_file() and bool(self.pid.read_text()),
          _TIMEOUT,
      )
    self.assertTrue(_gone(self._converter_pid(), 0))
    self.assertFalse(out.exists())

  def test_a_conversion_that_takes_too_long_is_stopped(self):
    """Past its timeout the child is killed, and the conversion fails."""
    self._fake(_SLEEPS)
    with self.assertRaisesRegex(
        RuntimeError,
        r'^The converter did not finish in 2 s, and was stopped\.$',
    ):
        graph_converter.convert(
            self.model,
            {'decode': 0},
            self.root / 'graphs.json',
            lambda: False,
            2,
        )
    self.assertTrue(_gone(self._converter_pid(), 0))

  def test_the_child_ends_when_its_parent_is_killed(self):
    """SIGKILL of the server's process: its converter ends too."""
    root = self._fake(_SLEEPS)
    parent = subprocess.Popen(
        [
            sys.executable,
            '-c',
            _PARENT,
            str(self.model),
            str(self.root / 'graphs.json'),
        ],
        env={
            **os.environ,
            'PYTHONPATH': os.pathsep.join([str(root), *sys.path]),
        },
    )
    self.addCleanup(_kill, parent.pid)
    child = self._converter_pid()
    parent.kill()
    parent.wait()
    self.assertTrue(_gone(child, 10), f'converter {child} outlived it')

  def test_a_converter_killed_by_a_signal_says_which(self):
    """A negative status is a signal, named."""
    self._fake(_SIGNALS)
    with self.assertRaisesRegex(
        RuntimeError,
        r'^The converter was killed by signal 15 \(SIGTERM\): ',
    ):
      graph_converter.convert(
          self.model,
          {'decode': 0},
          self.root / 'graphs.json',
          lambda: False,
          _TIMEOUT,
      )

  def test_a_missing_converter_is_not_available_here(self):
    """No wheel for this platform: the converter is not available."""
    self._fake(_MISSING)
    with self.assertRaises(RuntimeError) as raised:
      graph_converter.convert(
          self.model,
          {'decode': 0},
          self.root / 'graphs.json',
          lambda: False,
          _TIMEOUT,
      )
    self.assertTrue(
        str(raised.exception).startswith(graph_converter.UNAVAILABLE),
        raised.exception,
    )
    self.assertIn('_ffi', str(raised.exception))

  def test_a_failure_says_why(self):
    """A file that is not a .litertlm model: the child's words."""
    self.model.write_bytes(b'not a model' * 100)
    with self.assertRaisesRegex(RuntimeError, 'The converter failed'):
      graph_converter.convert(
          self.model,
          {'decode': 0},
          self.root / 'graphs.json',
          lambda: False,
          _TIMEOUT,
      )

  def test_the_real_model(self):
    """RUNNER_MODEL's graphs: ids, layers and anchors, as the scan's."""
    model = os.environ.get('RUNNER_MODEL')
    if not model:
      self.skipTest('RUNNER_MODEL names no model.')
    model = pathlib.Path(model)
    subgraphs = {s.name: s.subgraph for s in tap_scan.scan(model).signatures}
    out = self.root / 'graphs.json'
    graph_converter.convert(model, subgraphs, out, lambda: False, _TIMEOUT)
    graphs = json.loads(out.read_text())
    stems = model_graph.layer_stems(graphs).stems
    points = tap_scan.scan(model, stems).points
    # Every point is its node's output, tensor and name alike.
    found = 0
    for name, graph in graphs.items():
      nodes = {node['id']: node for node in graph['nodes']}
      for point in (p for p in points if p.signature == name):
        entry = nodes[str(point.op)]['outputsMetadata'][point.output]
        facts = model_graph.attrs(entry)
        self.assertEqual(entry['id'], str(point.output))
        self.assertEqual(facts['tensor_index'], str(point.tensor))
        self.assertEqual(facts['tensor_name'], point.tensor_name)
        found += 1
      model_graph.annotate(graph, points, name)
    self.assertLen(points, found)
    # Gemma 4's layers are `layer_<N>`, as tap_scan's old rule read them,
    # and every point is in the layer that rule put it in.
    self.assertEqual(set(stems.values()), {'layer'})
    for point in points:
      match = _OLD_LAYER.search(point.tensor_name.split(';')[0])
      self.assertEqual(
          point.layer, int(match.group(1)) if match else None, point.id
      )
    known = architecture.detect(graphs, stems, points)
    self.assertEqual(known.architecture.id, 'gemma4', known.text)
    document = known.architecture.skeleton(known.skeleton_id)
    bound = architecture.bind(
        known.architecture, document, graphs, stems, points
    )
    for name in ('decode', 'verify'):
      for entry in bound[name]:
        for anchor, binding in entry['anchors'].items():
          self.assertTrue(binding['bound'], (name, entry, anchor))
          self.assertTrue(binding['checked'], (name, entry, anchor))
    # The output head runs where the architecture says: only the decode
    # signatures write logits (LanguageModel.decode_softmax).
    (head,) = [
        m for m in known.architecture.outside() if m['id'] == 'output_head'
    ]
    self.assertEqual(head['stages'], ['decode'])
    self.assertEqual(
        {
            name
            for name, graph in graphs.items()
            if any(
                'LanguageModel.decode_softmax' in node.get('namespace', '')
                for node in graph['nodes']
            )
        },
        {name for name in graphs if not name.startswith('prefill')},
    )
    # T2's layer output is each layer's anchor `output`.
    for name, layers in bound.items():
      outputs = sorted(
          p.id for p in points if p.signature == name and p.layer_output
      )
      anchored = sorted(
          entry['anchors']['output']['point']
          for entry in layers
          if entry['anchors']['output']['bound']
      )
      self.assertEqual(outputs, anchored, name)


if __name__ == '__main__':
  test_main.main()
