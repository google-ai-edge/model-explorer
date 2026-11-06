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

"""Tests of the device catalog: device checks and capabilities."""

import pathlib

from absl.testing import absltest
from model_debugger.protocol import runner_pb2
from model_debugger.server.runner import app_version
from model_debugger.server.runner import launcher
from model_debugger.server.runner import runner_builds
from model_debugger.server.service import device_catalog
from model_debugger.server.service import option_catalog
from model_debugger.server.service import tap_view
from model_debugger.server.testing import fake_builds
from model_debugger.server.testing import fake_launchers
from model_debugger.server.testing import fake_runner
from model_debugger.server.testing import test_main
from model_debugger.server.testing import world as world_lib
from model_debugger.server.util import errors

_SIMULATOR = launcher.Device(
    'ios-simulator:S', 'iPhone 17', 'iOS Simulator', 'simulator', False
)
_OTHER_COMMIT = '0123456789abcdef0123456789abcdef01234567'
# Phones with their App: of the server's commit, of another, of none.
_PIXEL = launcher.Device(
    'android:P',
    'Pixel 10',
    'Android',
    'adb',
    False,
    app=app_version.App(True, app_version.LITERT_LM_COMMIT, '2026-10-15'),
)
_IPHONE = launcher.Device(
    'ios:I',
    'lzr-15',
    'iOS',
    'devicectl',
    False,
    app=app_version.App(True, _OTHER_COMMIT[:12]),
)
_BARE = launcher.Device(
    'android:N',
    'sdk_gphone64',
    'Android',
    'adb',
    False,
    app=app_version.App(installed=False),
)


class DeviceCatalogTest(absltest.TestCase):
  """Tests of the device catalog: device checks and capabilities."""

  def setUp(self):
    """An empty directory, and a computer and two devices."""
    super().setUp()
    self.tmp_path = pathlib.Path(self.create_tempdir().full_path)
    self.world = world_lib.create(self)

  def _world(self, devices: list[launcher.Device]) -> world_lib.World:
    made = world_lib.World(self.tmp_path, devices)
    self.addCleanup(made.close)
    return made

  def test_checking_a_device_launches_nothing(self):
    """Checking and listing every device launches no runner."""

    def launch(device_id, build=None):
      raise AssertionError(f'{device_id} was launched.')

    self.world.launchers.launch = launch
    for device in self.world.launchers.devices():
      self.world.catalog.inspect(device.id)
      self.world.catalog.builds(device.id)
    self.world.catalog.devices()
    self.world.catalog.capabilities()
    self.assertEqual(self.world.launchers.launched, [])

  def test_a_device_shows_what_the_editor_shows_and_no_more(self):
    """A device's view: when it was checked, and the editor's fields."""
    inspected = self.world.catalog.inspect(fake_launchers.PHONE.id)
    self.assertEqual(inspected['id'], fake_launchers.PHONE.id)
    self.assertTrue(inspected['checkedAt'])
    self.assertEqual(
        set(inspected),
        {
            'id',
            'checkedAt',
            'capability',
            'statusText',
            'choosable',
            'holder',
        },
    )

  def test_this_computer_offers_its_builds_and_a_phone_none(self):
    """This computer is 'not_running': the editor offers its builds."""
    local = self.world.catalog.inspect('local')
    self.assertEqual(local['runnerState'], 'not_running')
    self.assertNotIn('runnerState', self.world.catalog.inspect('android:A'))
    (listed,) = [
        device
        for device in self.world.catalog.devices()['devices']
        if device['id'] == 'local'
    ]
    self.assertEqual(listed['runnerState'], 'not_running')
    self.assertEqual(
        self.world.catalog.builds('android:A'),
        {
            'deviceId': 'android:A',
            'builds': [],
            'canLaunch': False,
            'reason': 'Pixel runs the Runner App installed on it.',
        },
    )
    with self.assertRaisesRegex(errors.NotFound, 'No device none'):
      self.world.catalog.builds('none')

  def test_a_phone_shows_the_version_of_its_app(self):
    """A phone's App: its LiteRT-LM commit and date, read unlaunched."""
    world = self._world([fake_launchers.LOCAL, _PIXEL, _IPHONE])
    commit = app_version.LITERT_LM_COMMIT
    expected = {
        'installed': True,
        'version': commit,
        'commitDate': '2026-10-15',
        'label': f'LiteRT-LM {commit[:12]} · 2026-10-15',
    }
    self.assertEqual(world.catalog.inspect(_PIXEL.id)['app'], expected)
    listed = {d['id']: d for d in world.catalog.devices()['devices']}
    self.assertEqual(listed[_PIXEL.id]['app'], expected)
    self.assertEqual(listed[_PIXEL.id]['statusText'], 'Connected')
    self.assertNotIn('app', listed['local'])
    self.assertEqual(
        world.catalog.builds(_PIXEL.id),
        {
            'deviceId': _PIXEL.id,
            'builds': [],
            'canLaunch': False,
            'reason': (
                'Pixel 10 runs the Runner App installed on it: '
                f'LiteRT-LM {commit[:12]} · 2026-10-15.'
            ),
            'app': expected,
        },
    )
    # An App of another commit says so, and may still be chosen.
    iphone = world.catalog.inspect(_IPHONE.id)
    self.assertEqual(
        iphone['app'],
        {
            'installed': True,
            'version': _OTHER_COMMIT[:12],
            'label': f'LiteRT-LM {_OTHER_COMMIT[:12]}',
            'notice': (
                f'Built against LiteRT-LM {_OTHER_COMMIT[:12]}; '
                f'this server is written for {commit[:12]}.'
            ),
        },
    )
    self.assertTrue(iphone['choosable']['allowed'])
    self.assertEqual(world.launchers.launched, [])

  def test_a_phone_without_the_app_cannot_be_chosen(self):
    """A phone without the App says so; a start on it is refused."""
    world = self._world([fake_launchers.LOCAL, _BARE])
    inspected = world.catalog.inspect(_BARE.id)
    self.assertEqual(inspected['statusText'], 'Connected · no Runner App')
    reason = 'No Runner App is installed on sdk_gphone64.'
    self.assertEqual(
        inspected['choosable'], {'allowed': False, 'reason': reason}
    )
    self.assertEqual(inspected['app']['installed'], False)
    self.assertEqual(inspected['app']['label'], 'No Runner App installed')
    self.assertEqual(world.catalog.builds(_BARE.id)['reason'], reason)
    record = world.create(ref='local', target=_BARE.id)
    refusal = world_lib.raised(world.model_servers.initialize, record['id'])
    self.assertIsInstance(refusal, errors.Conflict)
    self.assertEqual(str(refusal), f'Target: {reason}')
    self.assertEqual(world.launchers.launched, [])
    # It runs no runner: alone, it is no device to generate on.
    alone = world_lib.World(
        pathlib.Path(self.create_tempdir().full_path), [_BARE]
    )
    self.addCleanup(alone.close)
    self.assertFalse(alone.catalog.capabilities()['available'])

  def test_the_builds_of_this_computer_as_the_picker_shows_them(self):
    """Each build's label, its commit and CL, or why it cannot launch."""
    self.world.launchers.builds_now['local'] = [
        fake_launchers.runner_build(fake_builds.NEW),
        fake_launchers.runner_build(
            fake_builds.OLD, 'runner is missing. Keep it again.'
        ),
        fake_launchers.runner_build(runner_builds.DEVELOPMENT),
    ]
    catalog = self.world.catalog.builds('local')
    # No build ran here yet: each offers the server's table, and says so.
    for build in catalog['builds']:
      self.assertEqual(
          build.pop('capability'),
          device_catalog.runtime_capability(
              True, '', None, device_catalog.UNREPORTED_BUILD
          ),
      )
    self.assertEqual(
        catalog,
        {
            'deviceId': 'local',
            'builds': [
                {
                    'id': fake_builds.NEW,
                    'label': 'LiteRT-LM 222222222222 · 2026-10-12',
                    'version': fake_builds.NEW,
                    'cl': '998385386',
                    'platform': 'macOS',
                    'available': True,
                    'runtime': 'LiteRT-LM',
                    # The newest kept build available: the default.
                    'default': True,
                },
                {
                    'id': fake_builds.OLD,
                    'label': 'LiteRT-LM 111111111111 · 2026-10-12',
                    'version': fake_builds.OLD,
                    'cl': '998385386',
                    'platform': 'macOS',
                    'available': False,
                    'reason': 'runner is missing. Keep it again.',
                    'runtime': 'LiteRT-LM',
                    'default': False,
                },
                {
                    'id': 'dev',
                    'label': (
                        'Development build · LiteRT-LM '
                        + fake_launchers.COMMIT[:12]
                    ),
                    'version': fake_launchers.COMMIT,
                    'cl': '',
                    'platform': 'macOS',
                    'available': True,
                    'runtime': 'LiteRT-LM',
                    'default': False,
                },
            ],
            'canLaunch': True,
        },
    )
    self.assertEqual(self.world.launchers.launched, [])

  def test_no_launchable_build_says_how_to_keep_one(self):
    """Builds that cannot launch: the picker cannot launch, and says why."""
    self.world.launchers.builds_now['local'] = [
        fake_launchers.runner_build(fake_builds.OLD, 'runner is missing.')
    ]
    catalog = self.world.catalog.builds('local')
    self.assertFalse(catalog['canLaunch'])
    self.assertIn('cli:keep', catalog['reason'])

  def test_every_device_offers_the_cpu_and_a_gpu(self):
    """Only a launched runner says whether it has a GPU; a check cannot."""
    made = self._world([_SIMULATOR])
    capability = made.catalog.inspect(_SIMULATOR.id)['capability']
    self.assertEqual(capability['backends'], ['CPU', 'GPU'])
    self.assertEqual(
        made.catalog.capabilities()['runtimes'][0]['backends'],
        ['CPU', 'GPU'],
    )

  def test_a_held_device_is_busy_with_its_owner(self):
    """A phone a Model Server holds is busy; this computer never is."""
    record = self.world.create(ref='local', target='android:A')
    self.world.start(record)
    launched = len(self.world.launchers.launched)
    inspected = self.world.catalog.inspect('android:A')
    self.assertEqual(inspected['statusText'], 'Busy')
    self.assertEqual(
        inspected['holder'], f'In use by {record["name"]} · Target'
    )
    # This computer is never held; it says how many runners run there.
    local = self.world.catalog.inspect('local')
    self.assertEqual(local['statusText'], 'Connected · 1 runner running')
    self.assertLen(self.world.launchers.launched, launched)

  def test_a_held_device_cannot_be_chosen_as_a_start_refuses_it(self):
    """A held phone is refused with the reason a start on it is."""
    record = self.world.create(ref='local', target='android:A')
    self.world.start(record)
    inspected = self.world.catalog.inspect('android:A')
    self.assertEqual(inspected['statusText'], 'Busy')
    self.assertEqual(
        inspected['holder'], f'In use by {record["name"]} · Target'
    )
    choosable = inspected['choosable']
    self.assertFalse(choosable['allowed'])
    other = self.world.create(ref='local', target='android:A')
    refusal = world_lib.raised(self.world.model_servers.initialize, other['id'])
    self.assertIsInstance(refusal, errors.Conflict)
    self.assertEqual(str(refusal), f'Target: {choosable["reason"]}')
    # This computer runs any number of runners.
    local = self.world.catalog.inspect('local')
    self.assertEqual(local['choosable'], {'allowed': True, 'reason': ''})

  def test_a_free_device_may_be_chosen_with_its_capability(self):
    """A free device: connected, choosable, with LiteRT-LM's options."""
    inspected = self.world.catalog.inspect(fake_launchers.PHONE.id)
    self.assertEqual(inspected['statusText'], 'Connected')
    self.assertEqual(inspected['holder'], '')
    self.assertEqual(inspected['choosable'], {'allowed': True, 'reason': ''})
    capability = inspected['capability']
    # No runner ran there yet: the server's table alone, and it says so.
    self.assertEqual(
        capability,
        {
            **self.world.catalog.capabilities()['runtimes'][0],
            'notice': device_catalog.UNREPORTED_DEVICE,
        },
    )
    self.assertEqual(capability['backends'], ['CPU', 'GPU'])
    self.assertEqual(
        capability['session_options'],
        option_catalog.session_options(None),
    )
    self.assertEqual(
        capability['chat_options'], option_catalog.chat_options(None)
    )

  def test_a_device_offers_what_its_last_runner_reported(self):
    """Once a runner ran on a device, its schemas and host decide."""
    info = runner_pb2.RunnerInfo()
    info.CopyFrom(fake_runner.DEFAULT_INFO)
    info.host.ClearField('gpu')
    info.runtime.chat_options_schema = '{"type": "object"}'
    self.world.launchers.infos[fake_launchers.PHONE.id] = info
    record = self.world.create(ref=fake_launchers.PHONE.id)
    self.world.start(record)
    self.world.model_servers.close(record['id'])
    # Kept after the runner stopped, for that device alone.
    capability = self.world.catalog.inspect(fake_launchers.PHONE.id)[
        'capability'
    ]
    self.assertEqual(capability['backends'], ['CPU'])
    self.assertEqual(
        capability['chat_options'], option_catalog.chat_options(info)
    )
    self.assertNotIn('notice', capability)
    thinking = {o['key']: o for o in capability['chat_options']}['thinking']
    self.assertFalse(thinking['available'])
    other = self.world.catalog.inspect(fake_launchers.TABLET.id)
    self.assertEqual(
        other['capability']['backends'],
        ['CPU', 'GPU'],
    )

  def test_each_build_offers_what_its_own_last_runner_reported(self):
    """On this computer, what a build offers is that build's runner's."""
    info = runner_pb2.RunnerInfo()
    info.CopyFrom(fake_runner.DEFAULT_INFO)
    info.runtime.version = fake_launchers.COMMIT
    info.runtime.chat_options_schema = '{"type": "object"}'
    self.world.launchers.infos['local'] = info
    record = self.world.create()
    self.world.start(record)
    self.world.model_servers.close(record['id'])
    builds = {
        build['id']: build['capability']
        for build in self.world.catalog.builds('local')['builds']
    }
    # The build that ran: its runner's schemas, no notice.
    self.assertEqual(
        builds[fake_launchers.COMMIT],
        device_catalog.runtime_capability(True, '', info),
    )
    self.assertNotIn('notice', builds[fake_launchers.COMMIT])
    # Another build never ran: the table, and a notice that says so.
    self.assertEqual(
        builds[runner_builds.DEVELOPMENT],
        device_catalog.runtime_capability(
            True, '', None, device_catalog.UNREPORTED_BUILD
        ),
    )
    # The device's own, with no build chosen, is the table.
    self.assertEqual(
        self.world.catalog.inspect('local')['capability'],
        device_catalog.runtime_capability(
            True, '', None, device_catalog.CHOOSE_BUILD
        ),
    )

  def test_a_chats_sides_offer_their_builds_bounded_by_their_runs(self):
    """Each side: its device and build's offer; its reply its context."""
    info = runner_pb2.RunnerInfo()
    info.CopyFrom(fake_runner.DEFAULT_INFO)
    info.host.ClearField('gpu')
    info.runtime.chat_options_schema = '{"type": "object"}'
    self.world.launchers.infos[fake_launchers.PHONE.id] = info
    ran = self.world.create(target=fake_launchers.PHONE.id)
    self.world.start(ran)
    self.world.model_servers.close(ran['id'])
    record = self.world.create(target=fake_launchers.PHONE.id)
    sides = self.world.catalog.chat_capabilities(record['id'])
    self.assertCountEqual(sides, ['ref', 'target'])
    runs = {run['id']: run for run in record['runs']}
    self.assertEqual(
        sides['target'],
        device_catalog.runtime_capability(
            True,
            '',
            info,
            '',
            {'maxOutputTokens': int(runs['target']['contextLength'])},
            # Its model's own sampling, as the run before reported it.
            self.world.models.get(self.world.model.artifact).sampler_defaults,
        ),
    )
    for side in ('ref', 'target'):
      output = {o['key']: o for o in sides[side]['chat_options']}[
          'maxOutputTokens'
      ]
      self.assertEqual(output['maximum'], int(runs[side]['contextLength']))
    thinking = {o['key']: o for o in sides['target']['chat_options']}[
        'thinking'
    ]
    self.assertFalse(thinking['available'])
    with self.assertRaises(errors.NotFound):
      self.world.catalog.chat_capabilities('none')

  def test_kept_reads_the_last_listing_and_devices_lists_anew(self):
    """kept() never lists; devices() lists now and keeps the listing."""
    launchers = fake_launchers.FakeLaunchers([fake_launchers.LOCAL])
    listed = []
    devices = launchers.devices

    def counted():
      listed.append(1)
      return devices()

    launchers.devices = counted
    kept = device_catalog.DeviceList(launchers)
    self.assertEqual(kept.kept(), [])
    self.assertEqual(kept.devices(), [fake_launchers.LOCAL])
    launchers.devices_now = [fake_launchers.LOCAL, fake_launchers.PHONE]
    self.assertEqual(kept.kept(), [fake_launchers.LOCAL])
    self.assertLen(kept.devices(), 2)
    self.assertLen(kept.kept(), 2)
    self.assertEqual(listed, [1, 1])

  def test_started_the_list_is_listed_anew_every_few_seconds(self):
    """start() lists now, then on a thread of its own until stop()."""
    launchers = fake_launchers.FakeLaunchers([fake_launchers.LOCAL])
    kept = device_catalog.DeviceList(launchers, seconds=0.01)
    kept.start()
    self.addCleanup(kept.stop)
    self.assertEqual(kept.kept(), [fake_launchers.LOCAL])
    launchers.devices_now = [fake_launchers.PHONE]
    world_lib.eventually(lambda: kept.kept() == [fake_launchers.PHONE])
    kept.stop()
    launchers.devices_now = []
    self.assertEqual(kept.kept(), [fake_launchers.PHONE])

  def test_the_editor_and_checks_list_anew_and_capabilities_read_kept(self):
    """devices() and inspect() list anew; capabilities() reads the kept."""
    self.world.launchers.devices_now = []
    # Kept: the computer is still there for the capabilities.
    self.assertTrue(self.world.catalog.capabilities()['available'])
    with self.assertRaises(errors.NotFound):
      self.world.catalog.inspect(fake_launchers.LOCAL.id)
    self.assertFalse(self.world.catalog.capabilities()['available'])
    self.world.launchers.devices_now = [fake_launchers.LOCAL]
    self.assertEqual(
        [d['id'] for d in self.world.catalog.devices()['devices']],
        [fake_launchers.LOCAL.id],
    )
    self.assertTrue(self.world.catalog.capabilities()['available'])

  def test_an_unknown_device_is_not_found(self):
    """A device that is not connected cannot be checked."""
    with self.assertRaisesRegex(errors.NotFound, 'No device none'):
      self.world.catalog.inspect('none')

  def test_devices_lists_every_device_with_its_check(self):
    """devices() is the launcher's list, each as a check answers it."""
    listed = self.world.catalog.devices()['devices'][1]
    self.assertEqual(
        {key: listed[key] for key in ('id', 'name', 'platform', 'transport')},
        {
            'id': 'android:A',
            'name': 'Pixel',
            'platform': 'Android',
            'transport': 'adb',
        },
    )
    self.assertEqual(listed['statusText'], 'Connected')

  def test_capabilities_offer_the_litert_lm_options(self):
    """Capabilities need no runner: the table is the server's."""
    capabilities = self.world.catalog.capabilities()
    self.assertTrue(capabilities['available'])
    self.assertEqual(capabilities['platform'], 'macOS')
    runtime = capabilities['runtimes'][0]
    self.assertEqual(
        runtime['session_options'], option_catalog.session_options(None)
    )
    self.assertEqual(runtime['chat_options'], option_catalog.chat_options(None))
    self.assertEqual(runtime['backends'], ['CPU', 'GPU'])
    self.assertEqual(
        capabilities['models'],
        [{
            'artifact': self.world.model.artifact,
            'name': self.world.model.name,
            'runtime': 'LiteRT-LM',
            'size': self.world.model.size,
            'on_server': True,
        }],
    )
    self.assertEqual(capabilities['tap_profiles'], tap_view.profiles())

  def test_capabilities_without_devices_is_unavailable(self):
    """No device: generation is unavailable, with a reason."""
    capabilities = self._world([]).catalog.capabilities()
    self.assertFalse(capabilities['available'])
    self.assertTrue(capabilities['reason'])


if __name__ == '__main__':
  test_main.main()
