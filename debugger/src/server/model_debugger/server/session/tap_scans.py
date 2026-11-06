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

"""The scans of the models' tensors that can be observed (tap_scan).

A scan of a model is the same for the same bytes, so it is kept by the
model's SHA-256 for the server's life: in memory, since one takes about a
second. Scans run one at a time, on a thread of their own, as the editor asks
for them; saving a Session's points (session_store) and preparing a copy
(prepared_models) scan at once if they must. A scan that places the points in
layers, by the stems the model's graph names them by and the layers it groups
the operators named by none in (model_graphs), is kept apart, by those too.

A model imported without its file (model_store.Model.facts) has its scans
in its facts instead, as the computer it came from made them
(stored_files()): the plain one, SCAN, and the one by its graph's layers,
LAYERED_SCAN, with the stems and groups it was made by. A scan asked for by
other layers than those cannot be made without the file.
"""

from collections.abc import Mapping
import concurrent.futures
import dataclasses
import gzip
import json
import pathlib
import threading
from typing import Any

from absl import logging
from model_debugger.server.session import model_store
from model_debugger.server.tap import tap_scan

IDLE = 'idle'
SCANNING = 'scanning'
COMPLETED = 'completed'
FAILED = 'failed'
# The files of a model's facts that hold its scans (keep()).
SCAN = 'scan.json.gz'
LAYERED_SCAN = 'scan-layers.json.gz'


@dataclasses.dataclass(frozen=True)
class State:
  """Where the scan of a model is.

  Attributes:
      status: IDLE (never asked for), SCANNING, COMPLETED or FAILED.
      scan: The scan, once COMPLETED.
      error: Why it FAILED.
  """

  status: str
  scan: tap_scan.Scan | None = None
  error: str = ''


class TapScans:
  """Every model's scan, by SHA-256, kept until the server exits."""

  def __init__(self, models: model_store.ModelStore) -> None:
    """Holds no scans; close() stops the thread they run on."""
    self._models = models
    self._lock = threading.Lock()
    self._states: dict[str, State] = {}
    # (SHA-256, stems, groups) -> the scan with layers.
    self._layered: dict[tuple[Any, ...], tap_scan.Scan] = {}
    self._pool = concurrent.futures.ThreadPoolExecutor(
        1, thread_name_prefix='tap-scan'
    )

  def state(self, artifact: str) -> State:
    """Where the scan of a model is now.

    Args:
      artifact: The model artifact identifier.

    Returns:
      The current scan state of the model.

    Raises:
      NotFound: No such model.
    """
    model = self._models.get(artifact)
    with self._lock:
      return self._states.get(model.sha256, State(IDLE))

  def start(self, artifact: str) -> State:
    """Starts scanning a model, unless it is scanned or scanning.

    A scan that failed runs again.

    Args:
      artifact: The model artifact identifier.

    Returns:
      The scan state of the model.

    Raises:
      NotFound: No such model.
    """
    model = self._models.get(artifact)
    with self._lock:
      state = self._states.get(model.sha256, State(IDLE))
      if state.status in (SCANNING, COMPLETED):
        return state
      state = self._states[model.sha256] = State(SCANNING)
    self._pool.submit(self._run, model)
    return state

  def kept(self, model: model_store.Model) -> tap_scan.Scan | None:
    """The scan of a model, if one is kept; it scans nothing.

    The scan in a model's facts is kept too.

    Args:
      model: The model to get the scan for.

    Returns:
      The scan of the model if available, or None.
    """
    with self._lock:
      state = self._states.get(model.sha256)
    if state is not None:
      return state.scan
    try:
      found = self._stored(model)
    except (OSError, ValueError):
      return None
    if found is not None:
      with self._lock:
        self._states[model.sha256] = State(COMPLETED, found)
    return found

  def scan(
      self,
      model: model_store.Model,
      layer_stems: Mapping[str, str] | None = None,
      layer_groups: Mapping[str, Mapping[int, int]] | None = None,
  ) -> tap_scan.Scan:
    """The scan of a model: the one kept, or one made now.

    Args:
      model: The model.
      layer_stems: Per signature, the stem its layers are named by
        (tap_scan.scan()); without, no point is in a layer.
      layer_groups: Per signature, the layer of each operator whose outputs'
        names may name none (tap_scan.scan()).

    Returns:
      The model scan.

    Raises:
      ValueError: The model cannot be observed.
    """
    if layer_stems:
      key = _key(model.sha256, layer_stems, layer_groups)
      with self._lock:
        kept = self._layered.get(key)
      if kept is not None:
        return kept
      found = self._stored(model, key)
      if found is None:
        found = tap_scan.scan(_file(model), layer_stems, layer_groups)
      with self._lock:
        self._layered[key] = found
      return found
    with self._lock:
      state = self._states.get(model.sha256)
    if state is not None and state.scan is not None:
      return state.scan
    found = self._stored(model)
    if found is None:
      found = tap_scan.scan(_file(model))
    with self._lock:
      self._states[model.sha256] = State(COMPLETED, found)
    return found

  def stored_files(
      self,
      model: model_store.Model,
      layer_stems: Mapping[str, str] | None = None,
      layer_groups: Mapping[str, Mapping[int, int]] | None = None,
  ) -> dict[str, bytes]:
    """A model's scans as the files its facts hold them in, by name.

    The plain scan (SCAN), and, given the layers its graph names
    (model_graphs.ModelGraphs.layers()), the scan by them
    (LAYERED_SCAN): what reading a Session of the model asks for.

    Args:
      model: A LiteRT-LM model.
      layer_stems: As for scan().
      layer_groups: As for scan().

    Returns:
      A dictionary mapping scan filename to gzipped json bytes.

    Raises:
      ValueError, OSError: The model cannot be scanned.
    """
    files = {SCAN: _gzipped({'scan': tap_scan.to_json(self.scan(model))})}
    if layer_stems:
      files[LAYERED_SCAN] = _gzipped({
          'layer_stems': dict(layer_stems),
          'layer_groups': {
              name: sorted([op, layer] for op, layer in groups.items())
              for name, groups in (layer_groups or {}).items()
          },
          'scan': tap_scan.to_json(self.scan(model, layer_stems, layer_groups)),
      })
    return files

  def _stored(
      self, model: model_store.Model, key: tuple[Any, ...] | None = None
  ) -> tap_scan.Scan | None:
    """The scan in a model's facts: the plain one, or that of `key`.

    None for a model whose file the server reads.

    Args:
      model: The model to look up stored facts for.
      key: Optional scan cache key for layered scan.

    Returns:
      The stored scan, or None.

    Raises:
      ValueError: Its facts hold no such scan.
    """
    if model.facts is None or model.present():
      return None
    if key is None:
      return tap_scan.from_json(_read(model.facts / SCAN)['scan'])
    stored = _read(model.facts / LAYERED_SCAN)
    stems = stored['layer_stems']
    groups = {
        name: {int(op): int(layer) for op, layer in pairs}
        for name, pairs in stored['layer_groups'].items()
    }
    if _key(model.sha256, stems, groups) != key:
      raise ValueError(
          f'{model.name} is not on this computer, and its layers are '
          'not those it was scanned by where it came from.'
      )
    return tap_scan.from_json(stored['scan'])

  def close(self) -> None:
    """Stops scanning; a scan under way is abandoned."""
    self._pool.shutdown(wait=False, cancel_futures=True)

  def _run(self, model: model_store.Model) -> None:
    try:
      found = self._stored(model)
      state = State(COMPLETED, found or tap_scan.scan(_file(model)))
    except (ValueError, OSError) as error:
      state = State(FAILED, error=str(error))
    except (RuntimeError, LookupError, TypeError) as error:
      logging.exception('Scanning %s failed', model.name)
      state = State(FAILED, error=f'The model cannot be read: {error}')
    with self._lock:
      self._states[model.sha256] = state


def _key(
    sha256: str,
    layer_stems: Mapping[str, str],
    layer_groups: Mapping[str, Mapping[int, int]] | None,
) -> tuple[Any, ...]:
  """What a scan by layers is kept by."""
  return (
      sha256,
      tuple(sorted(layer_stems.items())),
      tuple(
          sorted(
              (name, tuple(sorted(groups.items())))
              for name, groups in (layer_groups or {}).items()
          )
      ),
  )


def _file(model: model_store.Model) -> pathlib.Path:
  """The model's file, to scan.

  Args:
    model: The model whose file path to return.

  Returns:
    The path to the model file.

  Raises:
    ValueError: It has none here.
  """
  if model.path is None:
    raise ValueError(f'{model.name} is not on this computer.')
  return model.path


def _gzipped(value: dict[str, Any]) -> bytes:
  return gzip.compress(
      json.dumps(value, separators=(',', ':')).encode('utf-8'), mtime=0
  )


def _read(path: pathlib.Path) -> dict[str, Any]:
  try:
    with gzip.open(path, 'rt', encoding='utf-8') as file:
      return json.load(file)
  except FileNotFoundError:
    raise ValueError(f'The facts of this model hold no {path.name}.') from None
