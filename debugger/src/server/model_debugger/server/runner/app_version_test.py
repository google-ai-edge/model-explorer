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

"""Tests of the Runner App's version, as a phone reports it."""

from absl.testing import absltest
from model_debugger.server.runner import app_version
from model_debugger.server.testing import test_main

_APP = 'com.google.ai.edge.modeldebugger.runner'
_SERVER = app_version.LITERT_LM_COMMIT
_OTHER = '0123456789abcdef0123456789abcdef01234567'
# What `adb shell dumpsys package` prints of the App, cut short.
_DUMPSYS = f"""Activity Resolver Table:
  Non-Data Actions:
      android.intent.action.MAIN:
        2c8f1a0 {_APP}/.RunnerActivity filter 9d1e2b3

Packages:
  Package [{_APP}] (5a3c9e1):
    appId=10215
    pkg=Package{{8f0b6d2 {_APP}}}
    versionCode=1 minSdk=31 targetSdk=35
    versionName={_SERVER} (2026-10-15)
    flags=[ HAS_CODE ALLOW_CLEAR_USER_DATA ]
"""


class AppVersionTest(absltest.TestCase):
  """Tests of the Runner App's version, as a phone reports it."""

  def test_the_servers_commit_has_no_notice(self):
    """An App of the server's commit: its commit and date, no notice."""
    app = app_version.App(True, _SERVER, '2026-10-15')
    self.assertEqual(app.label, f'LiteRT-LM {_SERVER[:12]} · 2026-10-15')
    self.assertEqual(app.notice, '')
    # An iPhone's App names the first 12 hex digits alone.
    self.assertEqual(app_version.App(True, _SERVER[:12]).notice, '')

  def test_another_commit_says_so(self):
    """An App of another commit says which, and the server's."""
    app = app_version.App(True, _OTHER)
    self.assertEqual(app.label, f'LiteRT-LM {_OTHER[:12]}')
    self.assertEqual(
        app.notice,
        f'Built against LiteRT-LM {_OTHER[:12]}; this server is written '
        f'for {_SERVER[:12]}.',
    )

  def test_a_version_of_no_commit_names_none(self):
    """An App built before its version named the commit names none."""
    app = app_version.App(True, version='1.0.0')
    self.assertEqual(
        app.label,
        'Runner App of no known LiteRT-LM version (version 1.0.0)',
    )
    self.assertIn('build and install the App again', app.notice)
    self.assertEqual(
        app_version.App(True).label,
        'Runner App of no known LiteRT-LM version',
    )

  def test_no_app(self):
    """A phone without the App says so."""
    app = app_version.App(installed=False)
    self.assertEqual(app.label, 'No Runner App installed')
    self.assertIn('Install the Runner App', app.notice)

  def test_dumpsys_gives_the_version_name(self):
    """The versionName of the App's own section names commit and date."""
    app = app_version.from_android_package(_APP, _DUMPSYS)
    self.assertEqual(app, app_version.App(True, _SERVER, '2026-10-15'))

  def test_dumpsys_without_the_package_is_no_app(self):
    """A package not installed has no section."""
    self.assertEqual(
        app_version.from_android_package(_APP, 'Unable to find package: x\n'),
        app_version.App(installed=False),
    )

  def test_a_version_name_of_no_commit_is_kept_as_it_is(self):
    """A versionName of no commit is shown; 'null' is none."""
    for version, kept in (('1.0', '1.0'), ('null', '')):
      dumpsys = (
          f'Packages:\n  Package [{_APP}] (1):\n    versionName={version}\n'
      )
      self.assertEqual(
          app_version.from_android_package(_APP, dumpsys),
          app_version.App(True, version=kept),
      )

  def test_another_packages_version_is_not_the_apps(self):
    """A versionName after the App's section is another package's."""
    dumpsys = (
        f'Packages:\n  Package [{_APP}] (1):\n    appId=1\n'
        '  Package [com.other] (2):\n    versionName=9.9\n'
    )
    self.assertEqual(
        app_version.from_android_package(_APP, dumpsys),
        app_version.App(True),
    )

  def test_ios_versions_give_the_date_and_the_commits_digits(self):
    """YYYY.MM.DD and the decimal of 12 hex digits, or the whole commit."""
    digits = str(int(_SERVER[:12], 16))
    self.assertEqual(
        app_version.from_ios_bundle('2026.10.15', digits),
        app_version.App(True, _SERVER[:12], '2026-10-15'),
    )
    self.assertEqual(
        app_version.from_ios_bundle('2026.10.15', digits, _SERVER),
        app_version.App(True, _SERVER, '2026-10-15'),
    )
    # A commit of leading zeros keeps them.
    self.assertEqual(
        app_version.from_ios_bundle('2026.10.15', '1').commit,
        '000000000001',
    )

  def test_ios_versions_of_an_older_app_name_no_commit(self):
    """An App built before its versions named the commit: '1.0.0' (1)."""
    self.assertEqual(
        app_version.from_ios_bundle('1.0.0', '1'),
        app_version.App(True, version='1.0.0'),
    )
    self.assertEqual(app_version.from_ios_bundle('2026.10.15', 'x').commit, '')


if __name__ == '__main__':
  test_main.main()
