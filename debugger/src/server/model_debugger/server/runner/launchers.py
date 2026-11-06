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

"""Every way this computer can launch runners, as one list of devices.

Four kinds of device:
- this computer: the command-line runner, on a port of its choosing;
- an Android device over adb: the App, reached through `adb forward`;
- a booted iOS simulator: the App, on the Mac's own loopback;
- a paired iOS device: the App, reached over the CoreDevice tunnel.

Above this package nothing tells the platforms apart; this computer is
known only by launcher.LOCAL_DEVICE_ID.

What each launcher starts is recorded (launch_records): stop_all() stops
whatever is left as the server exits, and the records kept in the workspace
let the next server stop what a killed one left (record_in()).
"""

import pathlib
import shutil
import time

from absl import logging
from model_debugger.server.runner import android_launcher
from model_debugger.server.runner import ios_device_launcher
from model_debugger.server.runner import ios_simulator_launcher
from model_debugger.server.runner import launch_records
from model_debugger.server.runner import launcher
from model_debugger.server.runner import local_launcher
from model_debugger.server.runner import runner_builds
from model_debugger.server.util import errors


class Launchers(launcher.Devices):
  """The devices this computer can launch runners on."""

  def __init__(self, builds: runner_builds.RunnerBuilds) -> None:
    """Finds the tools; a missing one simply contributes no devices.

    Args:
        builds: The command-line runner builds of this computer; with none, it
          is no device.
    """
    adb = android_launcher.find_adb()
    xcrun = shutil.which('xcrun')
    self._records = launch_records.LaunchRecords()
    records = self._records
    self._launchers: list[launcher.Launcher] = [
        local_launcher.LocalLauncher(builds, records)
    ]
    if adb is not None:
      self._launchers.append(android_launcher.AndroidLauncher(adb, records))
    if xcrun is not None:
      self._launchers.append(
          ios_simulator_launcher.IosSimulatorLauncher(xcrun, records)
      )
      self._launchers.append(
          ios_device_launcher.IosDeviceLauncher(xcrun, records)
      )

  def devices(self) -> list[launcher.Device]:
    """Returns every device reached now, phones without the App too."""
    found = []
    for place in self._launchers:
      found.extend(place.devices())
    return found

  def builds(self, device_id: str) -> list[runner_builds.Build]:
    """Returns the runner builds a Session may choose on a device.

    Args:
      device_id: The ID of the device.

    Raises:
        NotFound: No such device now.
    """
    place, device = self._find(device_id)
    return place.builds(device)

  def launch(
      self,
      device_id: str,
      build: str | None = None,
      model_dir: pathlib.Path | None = None,
  ) -> launcher.LaunchedRunner:
    """Starts a fresh runner on the device; returns once it accepts calls.

    Whatever runner of this server's App ran on the device before is
    stopped.

    Args:
        device_id: The Device's id.
        build: The build to launch on a device that chooses_build.
        model_dir: As launcher.Launcher.launch().

    Returns:
        The launched runner handle.

    Raises:
        NotFound: No such device now.
        RuntimeError: The runner did not start within 30 s, its build
          cannot be launched, the device has no Runner App, or a tool
          failed.
    """
    place, device = self._find(device_id)
    if device.without_app:
      raise RuntimeError(f'No Runner App is installed on {device.name}.')
    deadline = time.monotonic() + launcher.LAUNCH_SECONDS
    if model_dir is not None:
      launched = place.launch(device, deadline, build, model_dir)
    else:
      launched = place.launch(device, deadline, build)
    try:
      launched.wait_until_serving(deadline)
    except BaseException:
      launched.stop()
      raise
    logging.info(
        'Launched a runner (%s) on %s at %s',
        build or 'installed',
        device.name,
        launched.address,
    )
    return launched

  def record_in(self, directory: pathlib.Path) -> None:
    """Stops what a previous server recorded in `directory` and left.

    Each record is stopped by the launcher of its kind, and only if what
    it names still runs; a record no launcher here can stop (adb gone,
    say) is dropped. From then on, records each launch there.

    Args:
      directory: The directory containing leftover records.
    """
    for fields in self._records.keep_in(directory):
      kind = fields.get('kind')
      place = next(
          (place for place in self._launchers if place.KIND == kind),
          None,
      )
      if place is None:
        logging.warning('Cannot stop what a previous server left: %s', fields)
        continue
      try:
        place.stop_leftover(fields)
      except Exception as error:  # pylint: disable=broad-exception-caught
        # A tool, or a device gone.
        logging.warning(
            'Could not stop what a previous server left (%s): %s',
            fields,
            error,
        )

  def stop_all(self) -> None:
    """Stops every runner launched and not stopped; nothing launches after.

    A launch still starting is stopped too, and fails.
    """
    self._records.stop_all()

  def _find(self, device_id: str) -> tuple[launcher.Launcher, launcher.Device]:
    """The launcher of a device, and the device, listed now.

    Args:
      device_id: The ID of the device to find.

    Returns:
      A tuple of (launcher, device).

    Raises:
        NotFound: No such device now.
    """
    # Every launcher is asked, as devices() asks them.
    found = []
    for place in self._launchers:
      for device in place.devices():
        found.append((place, device))
    match = next((pair for pair in found if pair[1].id == device_id), None)
    if match is None:
      raise errors.NotFound(f'No device {device_id} is available.')
    return match
