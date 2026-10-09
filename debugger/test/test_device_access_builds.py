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

"""Synthetic device/build probes. These never SSH to or launch a real device."""

import json
from pathlib import Path
import plistlib
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from model_explorer_debugger.runtime import (
    device_access as access,
    runner_builds as builds,
    runner_launch as launcher,
)


def result(stdout='', returncode=0):
  return subprocess.CompletedProcess([], returncode, stdout, '')


class DeviceAccessBuildTests(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name)
    self.owner = dict(
        serverId='server-a', sessionId='session-a', runId='target'
    )

  def register_device(self, **extra):
    value = (
        dict(
            id='ssh:mac',
            name='Remote Mac',
            host='dev-mac',
            platform='macOS',
            runnerExecutable=(
                '/Applications/Runner.app/Contents/MacOS/ModelDebuggerMac'
            ),
        )
        | extra
    )
    (self.root / 'devices.json').write_text(
        json.dumps(dict(version=1, devices=[value]))
    )
    return value

  def register_build(self, **extra):
    value = (
        dict(
            id='CL-123', devices=['local'], kind='macos-app', protocolVersion=4
        )
        | extra
    )
    (self.root / 'runner-builds.json').write_text(
        json.dumps(dict(version=1, builds=[value]))
    )
    return value

  def app(self, metadata=None, provenance=None, executable='ModelDebuggerMac'):
    path = self.root / 'Runner "quoted" $(touch ignored).app'
    (path / 'Contents/MacOS').mkdir(parents=True)
    (path / 'Contents/Resources').mkdir()
    info = dict(
        CFBundleExecutable=executable,
        CFBundleVersion='1',
        ModelDebuggerBuildID='CL-123',
    )
    if metadata is not None:
      info = metadata
    (path / 'Contents/Info.plist').write_bytes(plistlib.dumps(info))
    binary = path / 'Contents/MacOS' / executable
    binary.write_bytes(b'synthetic executable')
    binary.chmod(0o700)
    if provenance is not None:
      (path / 'Contents/Resources/runtime-build.json').write_text(
          json.dumps(provenance)
      )
    return path

  def running(self, **extra):
    return dict(
        status='ready',
        runner=dict(
            protocolVersion=4,
            build=dict(id='CL-123'),
            lifecycle='waiting',
            controlConnected=False,
            busy=False,
        )
        | extra,
    )

  def test_registration_rejects_invalid_shapes_and_ssh_options(self):
    for payload in (
        [],
        dict(version=1, devices=[None]),
        dict(version=1, devices=[dict(id=3)]),
    ):
      (self.root / 'devices.json').write_text(json.dumps(payload))
      with self.assertRaises(ValueError):
        access.configurations(self.root)
    self.register_device(host='-oProxyCommand=bad')
    with self.assertRaises(ValueError):
      access.configurations(self.root)

  def test_ssh_arguments_round_trip_without_shell_interpolation(self):
    self.register_device()
    arguments = ['cat', '/Applications/R "x" $(touch /tmp/no); `id`/Info.plist']
    with patch.object(access.subprocess, 'run', return_value=result()) as run:
      access.command(self.root, 'ssh:mac', arguments)
    argv = run.call_args.args[0]
    self.assertEqual(
        argv[:4], ['ssh', '-oBatchMode=yes', '-oConnectTimeout=5', 'dev-mac']
    )
    self.assertEqual(shlex.split(argv[4]), arguments)
    self.assertNotIn('shell', run.call_args.kwargs)

  def test_ssh_unreachable_missing_binary_or_bad_snapshot_never_means_free(
      self,
  ):
    self.register_device()
    cases = (
        [result(returncode=255)],
        [result(), result(returncode=1)],
        [result(), result(), result(returncode=1)],
        [result(), result(), result('')],
        [result(), result(), result('changed process schema')],
    )
    for responses in cases:
      with (
          self.subTest(responses=responses),
          patch.object(access, 'command', side_effect=responses),
      ):
        self.assertEqual(
            access.runner_observation(self.root, 'ssh:mac')['runnerState'],
            'unknown',
        )

  def test_ssh_absence_requires_installed_runner_and_full_snapshot(self):
    self.register_device()
    with patch.object(
        access,
        'command',
        side_effect=[
            result(),
            result(),
            result(' 1 /sbin/launchd\n 23 /bin/ps\n'),
        ],
    ):
      self.assertEqual(
          access.runner_observation(self.root, 'ssh:mac')['runnerState'],
          'not_running',
      )
    with patch.object(
        access,
        'command',
        side_effect=[
            result(),
            result(),
            result(' 1 /sbin/launchd\n 23 /Other Build/ModelDebuggerMac\n'),
        ],
    ):
      self.assertEqual(
          access.runner_observation(self.root, 'ssh:mac')['runnerState'],
          'active',
      )

  def test_adb_access_does_not_claim_android_runner_support(self):
    self.register_device(
        id='adb:serial', platform='Android', runnerPackage='dev.example.runner'
    )
    with patch.object(access, 'command', return_value=result()) as command:
      observation = access.runner_observation(self.root, 'adb:serial')
    self.assertEqual(
        observation, dict(connectionState='connected', runnerState='unknown')
    )
    self.assertEqual(command.call_args.args[2], ['true'])

  def test_actual_local_build_uses_same_provenance_identity_as_native(self):
    commit, diff = 'a' * 40, 'b' * 64
    app = self.app(
        metadata=dict(
            CFBundleExecutable='ModelDebuggerMac', CFBundleVersion='1'
        ),
        provenance=dict(litertLMCommit=commit, nativeSourceDiffSHA256=diff),
    )
    identity = '1@' + commit + '+' + diff
    self.register_build(id=identity, path=str(app))
    with patch.object(builds.sys, 'platform', 'darwin'):
      self.assertTrue(
          builds.catalog(self.root, 'local')['builds'][0]['available']
      )
      self.register_build(id='1', path=str(app))
      self.assertFalse(
          builds.catalog(self.root, 'local')['builds'][0]['available']
      )

  def test_explicit_build_id_takes_precedence_and_catalog_hides_path(self):
    app = self.app(provenance=dict(litertLMCommit='a' * 40))
    self.register_build(path=str(app), label='CL 123')
    with patch.object(builds.sys, 'platform', 'darwin'):
      catalog = builds.catalog(self.root, 'local')
    self.assertTrue(catalog['canLaunch'])
    self.assertNotIn('path', catalog['builds'][0])
    self.assertNotIn('executable', catalog['builds'][0])

  def test_duplicate_build_for_overlapping_device_is_ambiguous(self):
    values = [
        dict(id='CL-123', devices=devices, kind='macos-app')
        for devices in (['local'], ['local', 'ssh:mac'])
    ]
    (self.root / 'runner-builds.json').write_text(
        json.dumps(dict(version=1, builds=values))
    )
    with self.assertRaisesRegex(ValueError, 'Duplicate'):
      builds.catalog(self.root, 'local')

  def test_ssh_catalog_verifies_installed_bundle_identity(self):
    self.register_device()
    value = self.register_build(
        devices=['ssh:mac'],
        kind='ssh-macos-app',
        path='/Applications/Runner.app',
    )
    info = dict(
        CFBundleExecutable='ModelDebuggerMac',
        CFBundleVersion='1',
        ModelDebuggerBuildID='CL-other',
    )
    with patch.object(
        access,
        'command',
        side_effect=[result(json.dumps(info)), result(), result(returncode=1)],
    ):
      available, _ = builds._available(self.root, 'ssh:mac', value)
    self.assertFalse(available)
    info['ModelDebuggerBuildID'] = 'CL-123'
    with patch.object(
        access,
        'command',
        side_effect=[result(json.dumps(info)), result(), result(returncode=1)],
    ):
      self.assertTrue(builds._available(self.root, 'ssh:mac', value)[0])

  def test_local_device_aliases_cannot_select_ambiguous_paths(self):
    values = [
        dict(id='CL-123', devices=devices, kind='macos-app')
        for devices in (['local'], ['macos:synthetic'])
    ]
    (self.root / 'runner-builds.json').write_text(
        json.dumps(dict(version=1, builds=values))
    )
    with patch.object(
        builds, '_local_aliases', return_value={'local', 'macos:synthetic'}
    ):
      with self.assertRaisesRegex(ValueError, 'aliases'):
        builds.catalog(self.root, 'local')

  def test_ssh_declared_executable_cannot_point_outside_selected_app(self):
    self.register_device()
    value = self.register_build(
        devices=['ssh:mac'],
        kind='ssh-macos-app',
        path='/Applications/Runner.app',
        executable='/tmp/other',
    )
    with patch.object(
        access,
        'command',
        return_value=result(
            json.dumps(
                dict(
                    CFBundleExecutable='ModelDebuggerMac',
                    ModelDebuggerBuildID='CL-123',
                )
            )
        ),
    ):
      self.assertFalse(builds._available(self.root, 'ssh:mac', value)[0])

  def test_ios_unknown_or_partial_process_schema_fails_closed(self):
    self.register_build(
        devices=['ios:device'],
        kind='ios-app',
        bundleId='dev.modeldebugger.runner',
    )
    for payload in (
        {},
        dict(runningProcesses=[]),
        dict(runningProcesses=[dict(pid=1)]),
        dict(runningProcesses=[dict(bundleIdentifier='other'), dict(pid=2)]),
    ):
      with (
          self.subTest(payload=payload),
          patch(
              'model_explorer_debugger.runtime.ios_device.devicectl',
              return_value=payload,
          ),
      ):
        self.assertEqual(
            builds.process_state(self.root, 'ios:device')['runnerState'],
            'unknown',
        )
    with patch(
        'model_explorer_debugger.runtime.ios_device.devicectl',
        return_value=dict(
            runningProcesses=[dict(bundleIdentifier='dev.modeldebugger.runner')]
        ),
    ):
      self.assertEqual(
          builds.process_state(self.root, 'ios:device')['runnerState'], 'active'
      )

  def test_ios_app_version_does_not_certify_a_cl_build(self):
    value = self.register_build(
        devices=['ios:device'],
        kind='ios-app',
        bundleId='dev.modeldebugger.runner',
    )
    apps = [
        dict(bundleIdentifier='dev.modeldebugger.runner', bundleVersion='1')
    ]
    with patch(
        'model_explorer_debugger.runtime.ios_device.devicectl',
        return_value=dict(apps=apps),
    ):
      self.assertFalse(builds._available(self.root, 'ios:device', value)[0])

  def test_ios_absence_requires_installed_application_evidence(self):
    self.register_build(
        devices=['ios:device'],
        kind='ios-app',
        bundleId='dev.modeldebugger.runner',
    )
    process_list = dict(
        runningProcesses=[dict(bundleIdentifier='com.apple.other')]
    )
    for apps in ({}, dict(apps=[])):
      with patch(
          'model_explorer_debugger.runtime.ios_device.devicectl',
          side_effect=[process_list, apps],
      ):
        self.assertEqual(
            builds.process_state(self.root, 'ios:device')['runnerState'],
            'unknown',
        )
    with patch(
        'model_explorer_debugger.runtime.ios_device.devicectl',
        side_effect=[
            process_list,
            dict(apps=[dict(bundleIdentifier='dev.modeldebugger.runner')]),
        ],
    ):
      self.assertEqual(
          builds.process_state(self.root, 'ios:device')['runnerState'],
          'not_running',
      )

  def test_local_empty_or_changed_process_output_is_unknown(self):
    self.register_build(path=str(self.app()))
    for stdout in ('', 'ModelDebuggerMac\n', ' 1 /sbin/launchd\n unknown\n'):
      with (
          self.subTest(stdout=stdout),
          patch.object(builds.subprocess, 'run', return_value=result(stdout)),
      ):
        self.assertEqual(
            builds.process_state(self.root, 'local')['runnerState'], 'unknown'
        )
    with patch.object(
        builds.subprocess,
        'run',
        return_value=result(' 1 /sbin/launchd\n 25 /bin/ps\n'),
    ):
      self.assertEqual(
          builds.process_state(self.root, 'local')['runnerState'], 'not_running'
      )

  def test_occupied_runner_rejected_even_without_selected_build(self):
    state = self.running(owner=self.owner | dict(serverId='another-server'))
    with (
        patch(
            'model_explorer_debugger.runtime.runner_discovery.inspect',
            return_value=state,
        ),
        patch.object(builds.subprocess, 'run') as launch,
    ):
      with self.assertRaisesRegex(ValueError, 'occupied'):
        launcher.ensure_started(self.root, dict(device='local'), self.owner)
    launch.assert_not_called()

  def test_unknown_unpaired_device_never_triggers_auto_launch(self):
    with (
        patch(
            'model_explorer_debugger.runtime.runner_discovery.inspect',
            return_value=dict(status='unpaired'),
        ),
        patch.object(builds.subprocess, 'run') as launch,
    ):
      with self.assertRaisesRegex(ValueError, 'Check the device'):
        launcher.ensure_started(
            self.root, dict(device='ios:device'), self.owner
        )
    launch.assert_not_called()

  def test_matching_running_runner_is_reused_without_launch(self):
    for selected in (None, 'CL-123'):
      with (
          patch(
              'model_explorer_debugger.runtime.runner_discovery.inspect',
              return_value=self.running(),
          ),
          patch.object(builds.subprocess, 'run') as launch,
      ):
        launcher.ensure_started(
            self.root, dict(device='local', runnerBuild=selected), self.owner
        )
      launch.assert_not_called()

  def test_ended_ios_shell_reusable_only_once_resources_and_owner_cleared(self):
    shell = self.running(
        lifecycle='ended', environment=dict(platform='iOS'), residentSessions=0
    )
    with (
        patch(
            'model_explorer_debugger.runtime.runner_discovery.inspect',
            return_value=shell,
        ),
        patch.object(builds.subprocess, 'run') as launch,
    ):
      launcher.ensure_started(self.root, dict(device='ios:device'), self.owner)
    launch.assert_not_called()
    for changed in (
        dict(environment=dict(platform='macOS')),
        dict(busy=True),
        dict(controlConnected=True),
        dict(residentSessions=1),
        dict(lifecycle='ending'),
        dict(owner=self.owner),
    ):
      state = shell | {'runner': shell['runner'] | changed}
      with (
          self.subTest(changed=changed),
          patch(
              'model_explorer_debugger.runtime.runner_discovery.inspect',
              return_value=state,
          ),
      ):
        with self.assertRaises(ValueError):
          launcher.ensure_started(
              self.root, dict(device='ios:device'), self.owner
          )

  def test_wrong_build_busy_and_incompatible_running_runner_are_rejected(self):
    for state in (
        self.running(build=dict(id='CL-other')),
        self.running(controlConnected=True),
        self.running(busy=True),
        self.running(protocolVersion=2),
    ):
      with (
          self.subTest(state=state),
          patch(
              'model_explorer_debugger.runtime.runner_discovery.inspect',
              return_value=state,
          ),
          patch.object(builds.subprocess, 'run') as launch,
      ):
        with self.assertRaises(ValueError):
          launcher.ensure_started(
              self.root, dict(device='local', runnerBuild='CL-123'), self.owner
          )
      launch.assert_not_called()

  def test_only_selected_verified_build_launches_and_metadata_is_rechecked(
      self,
  ):
    app = self.app()
    self.register_build(path=str(app))
    with (
        patch.object(builds.sys, 'platform', 'darwin'),
        patch(
            'model_explorer_debugger.runtime.runner_discovery.inspect',
            side_effect=[{}, self.running()],
        ),
        patch.object(
            builds,
            'process_state',
            return_value=dict(runnerState='not_running'),
        ),
        patch.object(builds.subprocess, 'run', return_value=result()) as launch,
    ):
      launcher.ensure_started(
          self.root, dict(device='local', runnerBuild='CL-123'), self.owner
      )
    self.assertEqual(
        launch.call_args.args[0],
        ['open', '-n', str(app), '--args', '--runner-session', 'session-a'],
    )
    self.assertNotIn('shell', launch.call_args.kwargs)

  def test_launch_refuses_unknown_process_state_and_unregistered_build(self):
    self.register_build(path=str(self.app()))
    for selected, presence in (
        ('unregistered', 'not_running'),
        ('CL-123', 'unknown'),
    ):
      with (
          patch.object(builds.sys, 'platform', 'darwin'),
          patch(
              'model_explorer_debugger.runtime.runner_discovery.inspect',
              return_value={},
          ),
          patch.object(
              builds, 'process_state', return_value=dict(runnerState=presence)
          ),
          patch.object(builds.subprocess, 'run') as launch,
      ):
        with self.assertRaises(ValueError):
          launcher.ensure_started(
              self.root, dict(device='local', runnerBuild=selected), self.owner
          )
      launch.assert_not_called()

  def test_ownership_race_after_launch_is_rejected(self):
    self.register_build(path=str(self.app()))
    with (
        patch.object(builds.sys, 'platform', 'darwin'),
        patch(
            'model_explorer_debugger.runtime.runner_discovery.inspect',
            side_effect=[
                {},
                self.running(owner=self.owner | dict(sessionId='other')),
            ],
        ),
        patch.object(
            builds,
            'process_state',
            return_value=dict(runnerState='not_running'),
        ),
        patch.object(builds.subprocess, 'run', return_value=result()),
    ):
      with self.assertRaisesRegex(ValueError, 'Another Session'):
        launcher.ensure_started(
            self.root, dict(device='local', runnerBuild='CL-123'), self.owner
        )


if __name__ == '__main__':
  unittest.main()
