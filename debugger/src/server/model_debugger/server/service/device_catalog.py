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

"""The devices runners can run on, as the UI lists and checks them.

Checking a device launches nothing: a device runs a runner only for the
Model Server that holds it. A held device is busy with its owner; any other
connected device is free for a Session. What a device offers is
`option_catalog.py`'s table of LiteRT-LM's options merged with what the last
runner there reported (ModelServers.reported()), each described for the UI;
on this computer, which runs the build a Session chooses, each build's own
(builds()). Before a runner of that build ran there the table alone decides,
and the capability's `notice` says so: the CPU and a GPU, and a GPU Load
fails if the runner reports none. What a Chat's sides may set is each side's
device and build's offer, bounded by its run (chat_capabilities()).

This computer launches the runner build each Session chooses (its
runnerState 'not_running' has the editor offer them, from builds()); a
phone's runner is the App installed on it, which no Session chooses yet:
its `app` says which LiteRT-LM the App was built against, as the phone
reports the App's version without launching it, or that none is installed.

Listing the devices runs a tool per platform (adb, xcrun), so the list is
kept (DeviceList). What a request the user makes for a device asks lists them
anew: the editor's device list, a device check, a start. What is read without
asking, the Session listing the UI polls and the capabilities, reads the kept
list, which a thread of its own lists anew every few seconds.
"""

import pathlib
import threading
from typing import Any

from absl import logging
from model_debugger.protocol import runner_pb2
from model_debugger.server.runner import app_version
from model_debugger.server.runner import launcher
from model_debugger.server.runner import runner_builds
from model_debugger.server.service import option_catalog
from model_debugger.server.service import tap_view
from model_debugger.server.session import hf_models
from model_debugger.server.session import model_server as model_server_lib
from model_debugger.server.session import model_servers as model_servers_lib
from model_debugger.server.session import model_store
from model_debugger.server.session import options
from model_debugger.server.session import session_store
from model_debugger.server.util import clock
from model_debugger.server.util import errors

# How often the kept list of the devices is listed anew.
DEVICE_LIST_SECONDS = 10.0
_RUNTIME = 'LiteRT-LM'
# How to keep a runner build, for the reasons that none can be launched.
_KEEP = (
    'keep one with `bazel run //model_debugger/runner/cli:keep` in src/runner'
)
# What a capability says when no runner of its build has run on the device.
UNREPORTED_DEVICE = (
    "No runner has run on this device yet: these are the server's defaults, "
    "until the runner's own replace them."
)
UNREPORTED_BUILD = (
    'This runner build has not run on this device yet: these are the '
    "server's defaults, until the runner's own replace them."
)
# Why no PyTorch run can start.
NO_PYTORCH_RUNNER = (
    'This server runs no PyTorch runner: start it with --pytorch_runner, or '
    'from a checkout with uv installed.'
)
CHOOSE_BUILD = (
    "These are the server's defaults: a runner build's own are shown once it "
    'is chosen and has run on this computer.'
)


class DeviceList(launcher.Devices):
  """The devices of the launchers, and the last listing of them, kept.

  devices() lists them now and keeps them; kept() reads the last listing
  and never lists. Once start()ed, a thread of its own lists them every
  few seconds, so the kept list is never older than that and no reader
  waits on a tool. Whether a runner launched on a device is still there
  is its connection's to say (runner_client), never this list's.
  """

  def __init__(
      self,
      launchers: launcher.Devices,
      seconds: float = DEVICE_LIST_SECONDS,
  ) -> None:
    """Keeps the devices of `launchers`; lists none until asked.

    Args:
        launchers: What lists the devices and launches runners.
        seconds: How often start()'s thread lists them anew.
    """
    self._launchers = launchers
    self._seconds = seconds
    # Held while listing, so that each listing replaces an older one.
    self._lock = threading.Lock()
    # Replaced whole by each listing, so kept() reads it without the lock.
    self._listed: list[launcher.Device] = []
    self._stopping = threading.Event()
    self._thread: threading.Thread | None = None

  def devices(self) -> list[launcher.Device]:
    """Lists the devices now, keeps them, and returns them."""
    with self._lock:
      self._listed = self._launchers.devices()
      return list(self._listed)

  def kept(self) -> list[launcher.Device]:
    """Returns the devices of the last listing; lists nothing."""
    return list(self._listed)

  def start(self) -> None:
    """Lists the devices now, then every few seconds until stop()."""
    self.devices()
    self._thread = threading.Thread(
        target=self._keep_listing, name='device-list', daemon=True
    )
    self._thread.start()

  def stop(self) -> None:
    """Stops listing; returns once a listing in progress ended."""
    self._stopping.set()
    if self._thread is not None:
      self._thread.join()

  def builds(self, device_id: str) -> list[runner_builds.Build]:
    """Returns the builds of a device, read anew, as the launchers do."""
    return self._launchers.builds(device_id)

  def record_in(self, directory: pathlib.Path) -> None:
    """As the launchers do."""
    self._launchers.record_in(directory)

  def stop_all(self) -> None:
    """As the launchers do."""
    self._launchers.stop_all()

  def launch(
      self,
      device_id: str,
      build: str | None = None,
      model_dir: pathlib.Path | None = None,
  ) -> launcher.LaunchedRunner:
    """Starts a fresh runner on a device, as the launchers do."""
    if model_dir is not None:
      return self._launchers.launch(device_id, build, model_dir)
    return self._launchers.launch(device_id, build)

  def _keep_listing(self) -> None:
    while not self._stopping.wait(self._seconds):
      try:
        self.devices()
      except Exception:  # pylint: disable=broad-exception-caught
        # The next listing tries again; the kept list stays.
        logging.exception('Listing the devices failed.')


def runtime_capability(
    available: bool,
    reason: str,
    info: runner_pb2.RunnerInfo | None = None,
    notice: str = '',
    maximums: dict[str, Any] | None = None,
    sampler_defaults: dict[str, Any] | None = None,
    runtime: str = _RUNTIME,
) -> dict[str, Any]:
  """The UI's RuntimeCapability of a runtime: its options, and if it runs.

  Args:
      available: Whether a device can run a runner.
      reason: Why not, if not.
      info: What the last runner on the device reported; None for the server's
        table alone.
      notice: What the UI shows with the options, when info is None: whose they
        are; '' for none.
      maximums: The highest value of a Chat setting its side's run bounds
        (options.chat_maximums), or None.
      sampler_defaults: The sampling of the side's model, as model_store.Model
        has it, or None.
      runtime: model_store.LITERT_LM or model_store.PYTORCH.

  Returns:
      The capability, its options described (option_catalog); a PyTorch
      one says what it observes and by which options (`tap`,
      option_catalog.pytorch_tap()).
  """
  capability = {
      'id': runtime,
      'available': available,
      'reason': reason,
      **option_catalog.runtime_options(
          info, maximums, sampler_defaults, runtime
      ),
  }
  if runtime == model_store.PYTORCH:
    capability['tap'] = option_catalog.pytorch_tap(info)
  # What the editor's Hugging Face form of the runtime says.
  capability['huggingface'] = hf_models.form(runtime)
  if info is None and notice:
    capability['notice'] = notice
  return capability


class DeviceCatalog:
  """Every device a runner can run on now, and who holds each."""

  def __init__(
      self,
      devices: DeviceList,
      models: model_store.ModelStore,
      sessions: session_store.SessionStore,
      model_servers: model_servers_lib.ModelServers,
  ) -> None:
    """Reads devices from `devices` and holders from `model_servers`."""
    self._devices = devices
    self._models = models
    self._sessions = sessions
    self._model_servers = model_servers

  def devices(self) -> dict[str, Any]:
    """The UI's DeviceCandidates: every device, listed anew."""
    return {
        'devices': [
            {
                'name': device.name,
                'platform': device.platform,
                'transport': device.transport,
                **self._view(device),
            }
            for device in self._devices.devices()
        ]
    }

  def inspect(self, device_id: str) -> dict[str, Any]:
    """Checks a device, listed anew, without launching anything.

    Args:
      device_id: The device to inspect.

    Returns:
      The UI's Partial<RunnerDevice>.

    Raises:
      NotFound: No such device is connected.
    """
    available = {device.id: device for device in self._devices.devices()}
    device = available.get(device_id)
    if device is None:
      raise errors.NotFound(f'No device {device_id} is connected.')
    return self._view(device)

  def builds(self, device_id: str) -> dict[str, Any]:
    """The runner builds a Session may choose on a device; launches none.

    Args:
      device_id: The device whose builds to list.

    Returns:
      The UI's RunnerBuildCatalog: each build with its label, whether
      it can be launched as far as its files tell, whether a side of
      its runtime runs it unless it chose another (`default`: the
      first available of its runtime, the newest kept LiteRT-LM build
      before the development one), and what a run of it
      may set (its `capability`, as the last runner of it here
      reported); what it reports is checked when a side starts on it.

    Raises:
      NotFound: No such device is connected.
    """
    available = {device.id: device for device in self._devices.devices()}
    device = available.get(device_id)
    if device is None:
      raise errors.NotFound(f'No device {device_id} is connected.')
    if not device.chooses_build:
      catalog = {
          'deviceId': device_id,
          'builds': [],
          'canLaunch': False,
          'reason': f'{device.name} runs the Runner App installed on it.',
      }
      if device.app is not None:
        catalog['app'] = _app_view(device.app)
        if device.app.installed:
          catalog['reason'] = (
              f'{device.name} runs the Runner App installed on it:'
              f' {device.app.label}.'
          )
        else:
          catalog['reason'] = model_server_lib.no_app(device)
      return catalog
    builds = self._devices.builds(device_id)
    # Each runtime's default: the first available in the list's order,
    # the newest kept LiteRT-LM build first, the development one last.
    defaults = {}
    for build in builds:
      if build.available:
        defaults.setdefault(build.runtime, build.id)
    catalog = {
        'deviceId': device_id,
        'builds': [
            {
                'id': build.id,
                'label': build.label,
                'version': build.commit,
                'cl': build.cl,
                'platform': device.platform,
                'available': build.available,
                **({'reason': build.reason} if build.reason else {}),
                'runtime': build.runtime,
                'default': defaults.get(build.runtime) == build.id,
                'capability': runtime_capability(
                    True,
                    '',
                    self._model_servers.reported(device_id, build.id),
                    UNREPORTED_BUILD,
                    runtime=build.runtime,
                ),
            }
            for build in builds
        ],
        'canLaunch': any(build.available for build in builds),
    }
    if not catalog['canLaunch']:
      catalog['reason'] = f'No runner build here can be launched: {_KEEP}.'
    return catalog

  def chat_capabilities(self, record_id: str) -> dict[str, Any]:
    """What each side of a Session's chat may set; launches nothing.

    Each side's is what the last runner of its build on its device
    reported, merged with the server's table, and bounded by its run:
    no reply longer than its context; with what the last runner to load
    its model reported of the model's own sampling.

    Args:
      record_id: The session record identifier.

    Returns:
      The UI's ChatCapabilities: a RuntimeCapability by side.

    Raises:
      NotFound: No such record.
    """
    record = self._sessions.get(record_id)
    capabilities = {}
    for run in record['runs']:
      build = run.get('runnerBuild')
      try:
        model = self._models.get(run.get('artifact', ''))
      except errors.NotFound:
        model = None
      capabilities[run['id']] = runtime_capability(
          True,
          '',
          self._model_servers.reported(run['device'], build),
          UNREPORTED_BUILD if build else UNREPORTED_DEVICE,
          options.chat_maximums(run),
          model.sampler_defaults if model else None,
          run.get('runtime') or _RUNTIME,
      )
    return capabilities

  def capabilities(self) -> dict[str, Any]:
    """The UI's RuntimeCapabilities, of the kept list of the devices.

    Its runtime's options are the server's table alone: what a device
    offers is that device's `capability` (see _view()).

    Returns:
      The UI's RuntimeCapabilities structure.
    """
    devices = self._devices.kept()
    local = next((d for d in devices if d.id == launcher.LOCAL_DEVICE_ID), None)
    available = any(not device.without_app for device in devices)
    pytorch = local is not None and any(
        build.id == runner_builds.PYTORCH and build.available
        for build in self._local_builds()
    )
    if available:
      reason = ''
    else:
      reason = (
          f'No device can run a runner: {_KEEP}, or connect a device '
          'with the Runner App.'
      )
    result = {
        'available': available,
        'reason': reason,
        'models': [
            {
                'artifact': model.artifact,
                'name': model.name,
                'runtime': model.runtime,
                'size': model.size,
                # Its file is on the server: a Session may load it.
                'on_server': model.present(),
                **({'source': model.source} if model.source else {}),
            }
            for model in self._models.list()
        ],
        'runtimes': [
            runtime_capability(available, reason),
            runtime_capability(
                pytorch,
                '' if pytorch else NO_PYTORCH_RUNNER,
                self._model_servers.reported(
                    launcher.LOCAL_DEVICE_ID, runner_builds.PYTORCH
                ),
                runtime=model_store.PYTORCH,
            ),
        ],
        'tap_anchors': tap_view.anchors_capture(),
        'tap_profiles': tap_view.profiles(),
        'upload_limit_bytes': model_store.UPLOAD_LIMIT_BYTES,
    }
    if local is not None:
      result['platform'] = local.platform
    return result

  def _local_builds(self) -> list[runner_builds.Build]:
    """This computer's builds; none while it is not listed."""
    try:
      return self._devices.builds(launcher.LOCAL_DEVICE_ID)
    except errors.NotFound:
      return []

  def _view(self, device: launcher.Device) -> dict[str, Any]:
    """The UI's RunnerDevice of a device: who holds it, what it offers.

    Its `capability` is what a run on it may set: LiteRT-LM's options
    merged with what the last runner there reported; on a device that
    chooses its build, the table alone, as each build's is in builds().
    `statusText`, `choosable` and `holder` are what the editor shows;
    `choosable` refuses a device another Model Server holds, or a phone
    without the Runner App, with the reason a start of a Session on it is
    refused. A phone's `app` is its App's version (_app_view()).
    A device that chooses its build has `runnerState` 'not_running': the
    editor then offers its builds(); it says no runner of its own runs
    there before a start, not how many Sessions' runners run there; a
    shared device's `statusText` says how many do.

    Args:
      device: The device to describe.

    Returns:
      The UI's RunnerDevice of the device.
    """
    runners = self._model_servers.runners(device.id)
    view = {'id': device.id, 'checkedAt': clock.now()}
    if device.app is not None:
      view['app'] = _app_view(device.app)
    if device.chooses_build:
      # No runner runs there until a start launches the build it chose;
      # what each build offers is in builds().
      view['runnerState'] = 'not_running'
      view['capability'] = runtime_capability(True, '', None, CHOOSE_BUILD)
    else:
      view['capability'] = runtime_capability(
          True,
          '',
          self._model_servers.reported(device.id),
          UNREPORTED_DEVICE,
      )
    if device.without_app:
      return {
          **view,
          'statusText': 'Connected · no Runner App',
          'choosable': {
              'allowed': False,
              'reason': model_server_lib.no_app(device),
          },
          'holder': '',
      }
    holding = self._model_servers.holding(device)
    if holding is None:
      # No Model Server holds it, or it is shared: a Session may use it.
      if runners:
        running_label = 'runner' if runners == 1 else 'runners'
        status_text = f'Connected · {runners} {running_label} running'
      else:
        status_text = 'Connected'
      return {
          **view,
          'statusText': status_text,
          'choosable': {'allowed': True, 'reason': ''},
          'holder': '',
      }
    try:
      owner = self._sessions.get(holding.owner_id)
      session_name = owner.get('name', '')
    except errors.NotFound:
      session_name = ''
    side = model_server_lib.SIDE_NAMES[holding.role]
    return {
        **view,
        'statusText': 'Busy',
        'choosable': {
            'allowed': False,
            'reason': model_server_lib.in_use(device),
        },
        'holder': f'In use by {session_name or holding.owner_id} · {side}',
    }


def _app_view(app: app_version.App) -> dict[str, Any]:
  """The UI's RunnerApp: the Runner App of a phone, as its version says.

  `version` is the LiteRT-LM commit it was built against (an iPhone's
  App names its first 12 hex digits alone), '' when its version names
  none; `notice`, when there is one, says why it may not be
  the App wanted: none installed, no commit named, or a commit other than
  the one this server is written for.

  Args:
    app: The app info parsed from the device.

  Returns:
    The UI's RunnerApp dict representation.
  """
  view = {
      'installed': app.installed,
      'version': app.commit,
      'label': app.label,
  }
  if app.commit_date:
    view['commitDate'] = app.commit_date
  if app.notice:
    view['notice'] = app.notice
  return view
