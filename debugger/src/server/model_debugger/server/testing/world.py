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

"""The service's layer in a temporary workspace, over fake launchers.

Usage:
  world = World(root, [fake_launchers.LOCAL])
  world.start(world.create())  # Its Model Server on, and a chat started.
  ...
  world.close()

or, in a test's setUp, the computer and two devices most tests use:
  self.world = world.create(self)
"""

import asyncio
import pathlib
import time
from typing import Any
from unittest import mock

from absl.testing import absltest
from model_debugger.server.runner import launcher
from model_debugger.server.service import device_catalog
from model_debugger.server.service import session_listing
from model_debugger.server.session import capture_store
from model_debugger.server.session import jobs
from model_debugger.server.session import mappings
from model_debugger.server.session import model_graphs
from model_debugger.server.session import model_servers
from model_debugger.server.session import model_store
from model_debugger.server.session import options
from model_debugger.server.session import prepared_models
from model_debugger.server.session import session_archive
from model_debugger.server.session import session_store
from model_debugger.server.session import tap_scans
from model_debugger.server.session import workspace
from model_debugger.server.testing import capture_fixtures
from model_debugger.server.testing import fake_converter
from model_debugger.server.testing import fake_launchers

_MODEL_BYTES = b'a model, as far as the fake runner can tell' * 1000
_LOCAL = launcher.LOCAL_DEVICE_ID
# The settings of a chat World starts: each side's defaults.
DEFAULT_CHAT = {'ref': {}, 'target': {}}
_DEFAULT_CHAT: Any = object()


class World:
  """Every store the Model Servers need, and what reads them."""

  def __init__(self, root, devices, graphs: bool = False) -> None:
    """Makes the stores under `root`, offering `devices`.

    Args:
        root: Where the workspace goes.
        devices: The devices offered.
        graphs: Whether the models' graphs are kept (model_graphs), converted by
          a fake_converter.FakeConverter, `converter`: a start converts them
          then, and the analyses place points in layers by them. Without,
          `graphs` is None.
    """
    # The fake runner's capture holds a tiny context, below the shortest
    # a real model is let have; until close(), Sessions may ask for it.
    self._tiny_context = mock.patch.object(
        options, 'MIN_CONTEXT_LENGTH', capture_fixtures.CONTEXT_LENGTH
    )
    self._tiny_context.start()
    self.workspace = workspace.Workspace(root / 'workspace')
    self.models = model_store.ModelStore(self.workspace)
    path = self.workspace.hf / 'gemma.litertlm'
    path.write_bytes(_MODEL_BYTES)
    self.model = self.models.add_download(path)
    self.scans = tap_scans.TapScans(self.models)
    self.sessions = session_store.SessionStore(
        self.workspace, self.models, self.scans
    )
    self.jobs = jobs.Jobs()
    self.converter = fake_converter.FakeConverter()
    if graphs:
      self.graphs = model_graphs.ModelGraphs(
          self.workspace,
          self.models,
          self.scans,
          self.jobs,
          convert=self.converter,
      )
    else:
      self.graphs = None
    self.launchers = fake_launchers.FakeLaunchers(devices)
    self.captures = capture_store.CaptureStore(self.workspace, self.sessions)
    # The Sessions being exported, which the listing shows.
    self.exports = session_archive.Exports()
    # Listed once, as the service starts them, but never by a thread of
    # its own: devices() lists them again after changing devices_now.
    self.devices = device_catalog.DeviceList(self.launchers)
    self.devices.devices()
    self.mappings = mappings.MappingStore(self.workspace)
    self.prepared = prepared_models.PreparedModels(
        self.workspace,
        self.models,
        self.sessions,
        self.jobs,
        self.scans,
        self.graphs,
        self.mappings,
    )
    self.model_servers = model_servers.ModelServers(
        self.sessions,
        self.models,
        self.jobs,
        self.devices,
        self.captures,
        self.prepared,
        self.graphs,
    )
    self.catalog = device_catalog.DeviceCatalog(
        self.devices, self.models, self.sessions, self.model_servers
    )
    self.listing = session_listing.SessionListing(
        self.sessions,
        self.model_servers,
        self.jobs,
        self.captures,
        self.prepared,
        self.exports,
    )

  def config(self, ref: str = _LOCAL, target: str = _LOCAL) -> dict[str, Any]:
    """Returns what the UI's editor posts for a Session on two devices.

    A side on this computer chooses the kept build of
    fake_launchers.COMMIT; one on a phone runs the App installed. Every
    Session made of it captures the anchors (session_store); older()
    makes one as a record written before that was.

    Args:
      ref: Device identifier for the reference run.
      target: Device identifier for the target run.

    Returns:
      Session configuration dictionary.
    """
    return {
        'name': 'CPU vs CPU',
        'model': 'Gemma',
        'runs': [
            {
                'id': side,
                'runtime': 'LiteRT-LM',
                'artifact': self.model.artifact,
                'source': 'upload',
                'device': device,
                'backend': 'CPU',
                'contextLength': str(capture_fixtures.CONTEXT_LENGTH),
                **(
                    {'runnerBuild': fake_launchers.COMMIT}
                    if device == _LOCAL
                    else {}
                ),
            }
            for side, device in (('ref', ref), ('target', target))
        ],
    }

  def create(self, ref: str = _LOCAL, target: str = _LOCAL) -> dict[str, Any]:
    """Creates a Session on two devices."""
    return self.sessions.create(self.config(ref, target))

  def older(self, record: dict[str, Any]) -> dict[str, Any]:
    """Makes a record one written before every Session captured anchors.

    Such a record captures none: its `tap_anchors` false, as it holds;
    set_flags() writes the fields it is given.

    Args:
      record: The Session record dictionary.

    Returns:
      The record.
    """
    self.sessions.set_flags(record['id'], tap_anchors=False)
    return self.sessions.get(record['id'])

  def wait(self, job: dict[str, Any]) -> dict[str, Any]:
    """Returns the job once it ended."""
    deadline = time.monotonic() + fake_launchers.TIMEOUT
    while time.monotonic() < deadline:
      job = self.jobs.get(job['id'])
      if job['status'] in jobs.TERMINAL:
        return job
      time.sleep(0.01)
    raise AssertionError(f'job {job["id"]} did not end: {job}')

  def start(
      self,
      record: dict[str, Any],
      chat: Any = _DEFAULT_CHAT,
  ) -> dict[str, Any]:
    """Turns a Session's Model Server on and starts a chat.

    Args:
      record: The Session.
      chat: The chat's settings, `{ref, target}`; each side's defaults by
        default. None starts no chat.

    Returns:
      The initialize job, completed.
    """
    job = self.wait(self.model_servers.initialize(record['id']))
    assert job['status'] == 'completed', job['error']
    if chat is _DEFAULT_CHAT:
      chat = DEFAULT_CHAT
    if chat is not None:
      self.chat(record['id'], chat)
    return job

  def chat(
      self,
      record_id: str,
      chat: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    """Starts a chat; returns its job, completed."""
    if chat is None:
      chat = DEFAULT_CHAT
    job = self.wait(self.model_servers.start_chat(record_id, chat))
    assert job['status'] == 'completed', job['error']
    return job

  def turn(self, record_id: str, prompt: str) -> dict[str, Any]:
    """Sends a turn; returns its job once it ended."""
    return self.wait(self.model_servers.generate(record_id, prompt))

  def summary(self, record_id: str) -> dict[str, Any]:
    """Returns a record as the listing shows it now."""
    return self.listing.summary(self.sessions.get(record_id))

  def view(self, record_id: str) -> dict[str, Any] | None:
    """Returns the listing's execution of a Session, if it ever ran."""
    return self.summary(record_id).get('execution')

  def scripts(self, device_id: str = _LOCAL) -> list[fake_launchers.Script]:
    """The scripts of the runners running on a device now."""
    return [launched.script for launched in self.launchers.running(device_id)]

  def events(self, job_id: str) -> list[dict[str, Any]]:
    """Returns every event of a job that ended, as a reader gets them."""

    async def read() -> list[dict[str, Any]]:
      return [event async for event in self.jobs.events(job_id, 0)]

    return asyncio.run(asyncio.wait_for(read(), fake_launchers.TIMEOUT))

  def close(self) -> None:
    """Stops every runner and releases the workspace."""
    self.model_servers.shutdown()
    self.scans.close()
    if self.graphs is not None:
      self.graphs.close()
    self.workspace.close()
    self._tiny_context.stop()


def create(test: absltest.TestCase, graphs: bool = False) -> World:
  """A computer and two devices that run one runner at a time, for `test`.

  The World lives in a directory of its own and closes when the test ends;
  `graphs` as World's.

  Args:
    test: The test case.
    graphs: Whether the models' graphs are kept.

  Returns:
    A newly created World instance.
  """
  made = World(
      pathlib.Path(test.create_tempdir().full_path),
      [fake_launchers.LOCAL, fake_launchers.PHONE, fake_launchers.TABLET],
      graphs,
  )
  test.addCleanup(made.close)
  return made


def eventually(condition) -> None:
  """Waits for a fake runner's handler to see what the client did."""
  deadline = time.monotonic() + fake_launchers.TIMEOUT
  while not condition():
    assert time.monotonic() < deadline
    time.sleep(0.01)


def raised(call, *args):
  """Returns what `call(*args)` raised, or None."""
  try:
    call(*args)
  except Exception as error:  # pylint: disable=broad-exception-caught
    return error
  return None
