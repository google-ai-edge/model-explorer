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

"""The Runner App installed on a phone, as its version says.

A phone runs the App installed on it, one version per phone. The App's build
writes the LiteRT-LM commit it is built against, and that commit's date, into
the App's own version (src/runner's app/), which a launcher reads without
launching the App:
- Android: versionName is `<commit> (<YYYY-MM-DD>)`, as adb's
  `dumpsys package` shows it;
- iOS: Apple takes numbers alone, so CFBundleShortVersionString is the date,
  YYYY.MM.DD, and CFBundleVersion the first 12 hex digits of the commit, in
  decimal, as devicectl lists them; the whole commit is the bundle's
  LiteRTLMCommit, which a simulator's copy of the App shows.
What the App's runner reports once it runs, GetInfo's runtime version, is
the same commit.
"""

import dataclasses
import re

# The LiteRT-LM commit this server is written for: the Runner's
# (src/runner/MODULE.bazel's LITERT_LM_COMMIT), whose options and schemas it
# knows (ci/check_schema_pins.py checks that they are the same).
LITERT_LM_COMMIT = '61dcfc77225019e067580dceb3ea268483d1938c'

# How many hex digits of the commit an iOS App's CFBundleVersion holds.
_IOS_DIGITS = 12
# Android's versionName: '<commit> (<commit date>)'.
_ANDROID_VERSION = re.compile(
    r'(?P<commit>[0-9a-f]{40}) \((?P<date>\d{4}-\d\d-\d\d)\)'
)
# iOS's CFBundleShortVersionString: the commit's date.
_IOS_DATE = re.compile(r'(?P<year>\d{4})\.(?P<month>\d\d)\.(?P<day>\d\d)')
_COMMIT = re.compile(r'[0-9a-f]{40}')


@dataclasses.dataclass(frozen=True)
class App:
  """The Runner App on a phone, as read without launching it.

  Attributes:
      installed: Whether the App is installed there.
      commit: The LiteRT-LM commit it was built against: whole, or its first 12
        hex digits where only they are known (an iPhone's App); '' when its
        version names none (an App built before it did).
      commit_date: When that commit was made, YYYY-MM-DD; '' if not known.
      version: Its version as the phone reports it, for one that names no
        commit; '' otherwise.
  """

  installed: bool
  commit: str = ''
  commit_date: str = ''
  version: str = ''

  @property
  def label(self) -> str:
    """What the user knows it by: 'LiteRT-LM 61dcfc772250 · 2026-10-15'."""
    if not self.installed:
      return 'No Runner App installed'
    if not self.commit:
      return 'Runner App of no known LiteRT-LM version' + (
          f' (version {self.version})' if self.version else ''
      )
    date = self.commit_date
    return f'LiteRT-LM {self.commit[:12]}' + (f' · {date}' if date else '')

  @property
  def notice(self) -> str:
    """Why it may not be the App this server is written for; '' if it is."""
    if not self.installed:
      return (
          'Install the Runner App (src/runner/README.md), then check the '
          'device again.'
      )
    if not self.commit:
      return (
          'Its version names no LiteRT-LM commit: build and install the '
          'App again from src/runner to see it here.'
      )
    if not LITERT_LM_COMMIT.startswith(self.commit):
      return (
          f'Built against LiteRT-LM {self.commit[:12]}; this server is '
          f'written for {LITERT_LM_COMMIT[:12]}.'
      )
    return ''


def from_android_package(app_id: str, dumpsys: str) -> App:
  """Parses the App as `adb shell dumpsys package <app_id>` describes it.

  Args:
    app_id: The App's package.
    dumpsys: The command's output: the package's `Package [<app_id>]` section,
      with its `versionName=`; none when it is not installed.

  Returns:
    The App parsed from the dumpsys output.
  """
  in_package = False
  for raw_line in dumpsys.splitlines():
    line = raw_line.strip()
    if not in_package:
      if line.startswith(f'Package [{app_id}]'):
        in_package = True
      continue
    if line.startswith('Package ['):
      break  # another package's section
    if line.startswith('versionName='):
      version = line.removeprefix('versionName=')
      found = _ANDROID_VERSION.fullmatch(version)
      if found:
        return App(True, found.group('commit'), found.group('date'))
      # An App whose manifest names no version: 'versionName=null'.
      return App(True, version='' if version == 'null' else version)
  if not in_package:
    return App(installed=False)
  return App(True)


def from_ios_bundle(
    short_version: str, bundle_version: str, commit: str = ''
) -> App:
  """The App as its bundle's versions describe it.

  Args:
    short_version: Its CFBundleShortVersionString: the commit's date.
    bundle_version: Its CFBundleVersion: the commit's first 12 hex digits, in
      decimal.
    commit: Its LiteRTLMCommit, the whole commit, where known.

  Returns:
    The App parsed from the bundle version info.
  """
  date = _IOS_DATE.fullmatch(short_version)
  if date is None:
    # An App built before its version named the commit: '1.0.0'.
    return App(True, version=short_version)
  commit_date = '{}-{}-{}'.format(*date.group('year', 'month', 'day'))
  if _COMMIT.fullmatch(commit):
    return App(True, commit, commit_date)
  if not bundle_version.isdigit() or int(bundle_version) >= 16**_IOS_DIGITS:
    return App(True, version=f'{short_version} ({bundle_version})')
  return App(True, f'{int(bundle_version):0{_IOS_DIGITS}x}', commit_date)
