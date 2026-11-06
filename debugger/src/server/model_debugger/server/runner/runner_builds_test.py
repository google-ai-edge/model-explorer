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

"""Tests of this computer's runner builds: kept ones and the development one."""

import json
import os
import pathlib

from absl.testing import absltest
from model_debugger.server.runner import runner_builds
from model_debugger.server.testing import fake_builds
from model_debugger.server.testing import test_main

_DEV = runner_builds.DEVELOPMENT


class RunnerBuildsTest(absltest.TestCase):
  """Tests of this computer's runner builds."""

  def setUp(self):
    """An empty kept builds' directory."""
    super().setUp()
    self.tmp_path = pathlib.Path(self.create_tempdir().full_path)
    self.kept = self.tmp_path / 'kept'

  def _builds(self, development=None) -> runner_builds.RunnerBuilds:
    return runner_builds.RunnerBuilds(self.kept, development)

  def test_kept_builds_are_read_newest_commit_first(self):
    """Each build's commit, date and CL, from its build.json."""
    old = fake_builds.keep(
        self.kept, fake_builds.OLD, commit_date='2026-10-12T22:51:09-07:00'
    )
    fake_builds.keep(
        self.kept,
        fake_builds.NEW,
        commit_date='2026-10-15T00:06:55-07:00',
        cl='999909015',
    )
    builds = self._builds().list()
    self.assertEqual(
        [build.id for build in builds], [fake_builds.NEW, fake_builds.OLD]
    )
    self.assertEqual(
        [build.label for build in builds],
        [
            'LiteRT-LM 222222222222 · 2026-10-15',
            'LiteRT-LM 111111111111 · 2026-10-12',
        ],
    )
    newest = builds[0]
    self.assertEqual(newest.commit, fake_builds.NEW)
    self.assertEqual(newest.cl, '999909015')
    self.assertTrue(newest.available)
    self.assertEqual(builds[1].runner, old / 'runner')
    self.assertEqual(set(builds[1].sha256), {'runner', *fake_builds.LIBRARIES})

  def test_kept_builds_are_ordered_by_when_their_commits_were_made(self):
    """Dates of other offsets compare as times, not as text."""
    # 2026-10-14T16:00Z, though its text sorts after the other's.
    fake_builds.keep(
        self.kept, fake_builds.OLD, commit_date='2026-10-15T01:00:00+09:00'
    )
    # 2026-10-15T03:00Z.
    fake_builds.keep(
        self.kept, fake_builds.NEW, commit_date='2026-10-14T20:00:00-07:00'
    )
    undated = 'a' * 40
    fake_builds.keep(self.kept, undated, commit_date='')
    self.assertEqual(
        [build.id for build in self._builds().list()],
        [fake_builds.NEW, fake_builds.OLD, undated],
    )

  def test_a_missing_directory_keeps_nothing(self):
    """No directory, or None: no kept build."""
    self.assertEqual(self._builds().list(), [])
    self.assertEqual(runner_builds.RunnerBuilds(None, None).list(), [])

  def test_what_is_not_a_kept_build_is_skipped(self):
    """Hidden staging, .partial and other names are no builds."""
    fake_builds.keep(self.kept, fake_builds.OLD)
    (self.kept / f'.keep.{fake_builds.NEW[:6]}').mkdir()
    (self.kept / f'{fake_builds.NEW}.partial').mkdir()
    (self.kept / 'notes').mkdir()
    (self.kept / fake_builds.NEW).write_text('a file')
    self.assertEqual(
        [build.id for build in self._builds().list()], [fake_builds.OLD]
    )

  def test_a_broken_build_is_listed_with_why(self):
    """A build.json that cannot be read, or a missing file: unavailable."""
    broken = fake_builds.keep(self.kept, fake_builds.OLD)
    (broken / 'build.json').write_text('{')
    missing = fake_builds.keep(self.kept, fake_builds.NEW)
    (missing / 'liblitert-lm.dylib').unlink()
    builds = {build.id: build for build in self._builds().list()}
    self.assertFalse(builds[fake_builds.OLD].available)
    self.assertRegex(
        builds[fake_builds.OLD].reason, r'^build.json cannot be read'
    )
    self.assertEqual(
        builds[fake_builds.NEW].reason,
        'liblitert-lm.dylib is missing. Keep it again.',
    )
    # Its label still names the commit, and the date it recorded.
    self.assertEqual(
        builds[fake_builds.NEW].label, 'LiteRT-LM 222222222222 · 2026-10-12'
    )

  def test_a_build_json_of_another_commit_or_without_sums_is_unavailable(
      self,
  ):
    """build.json must name its directory's commit and the runner's sum."""
    for commit, change, reason in (
        (fake_builds.OLD, {'commit': fake_builds.NEW}, 'names LiteRT-LM'),
        (fake_builds.NEW, {'sha256': {}}, 'no SHA-256 of the runner'),
    ):
      path = fake_builds.keep(self.kept, commit) / 'build.json'
      path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
      (build,) = [b for b in self._builds().list() if b.id == commit]
      self.assertIn(reason, build.reason)

  def test_a_build_json_must_list_every_library_kept(self):
    """A library build.json does not list would run unchecked."""
    path = fake_builds.keep(self.kept, fake_builds.OLD) / 'build.json'
    facts = json.loads(path.read_text())
    del facts['sha256']['libwebgpu_dawn.dylib']
    path.write_text(json.dumps(facts))
    (build,) = self._builds().list()
    self.assertEqual(
        build.reason,
        'build.json lists no SHA-256 of libwebgpu_dawn.dylib. Keep it again.',
    )
    with self.assertRaisesRegex(RuntimeError, 'cannot be launched'):
      self._builds().runner(fake_builds.OLD)

  def test_a_library_or_program_it_was_not_kept_with_is_refused(self):
    """Any other dylib or executable file beside the runner: unavailable."""
    directory = fake_builds.keep(self.kept, fake_builds.OLD)
    (directory / 'notes.txt').write_text('a note')
    (directory / 'sub').mkdir()
    self.assertTrue(self._builds().list()[0].available)
    extra = directory / 'libextra.dylib'
    extra.write_bytes(b'a library')
    (build,) = self._builds().list()
    self.assertEqual(
        build.reason,
        'libextra.dylib is not one of the files it was kept with. Keep it '
        'again.',
    )
    with self.assertRaisesRegex(RuntimeError, 'cannot be launched'):
      self._builds().runner(fake_builds.OLD)
    extra.unlink()
    tool = directory / 'tool'
    tool.write_text('#!/bin/sh\n')
    tool.chmod(0o755)
    (build,) = self._builds().list()
    self.assertEqual(
        build.reason,
        'tool is not one of the files it was kept with. Keep it again.',
    )

  def test_runner_checks_every_file_before_it_is_run(self):
    """A kept build runs once each file has the SHA-256 it was kept with."""
    directory = fake_builds.keep(self.kept, fake_builds.OLD)
    builds = self._builds()
    self.assertEqual(builds.runner(fake_builds.OLD), directory / 'runner')
    library = directory / 'liblitert-lm.dylib'
    library.write_bytes(b'X' + library.read_bytes()[1:])
    with self.assertRaises(RuntimeError) as changed:
      builds.runner(fake_builds.OLD)
    self.assertEqual(
        str(changed.exception),
        'LiteRT-LM 111111111111 · 2026-10-12 was changed after it was '
        'kept: liblitert-lm.dylib does not match the SHA-256 in its '
        'build.json. Keep it again.',
    )
    library.unlink()
    with self.assertRaisesRegex(RuntimeError, 'cannot be launched: lib'):
      builds.runner(fake_builds.OLD)

  def test_runner_refuses_a_build_no_longer_there(self):
    """A build removed, or none chosen, is refused with what to do."""
    builds = self._builds(self.tmp_path / 'absent')
    with self.assertRaises(runner_builds.NotThereError) as refused:
      builds.runner(fake_builds.NEW)
    self.assertEqual(
        str(refused.exception),
        'The runner for LiteRT-LM 222222222222 is no longer kept on this '
        'computer; keep it again.',
    )
    self.assertEqual(refused.exception.fix, 'keep it again')
    with self.assertRaises(runner_builds.NotThereError) as development:
      builds.runner(_DEV)
    self.assertEqual(
        str(development.exception),
        'The development build of the runner is no longer on this '
        'computer; build it again.',
    )
    with self.assertRaises(runner_builds.NotThereError) as unchosen:
      builds.runner('')
    self.assertEqual(str(unchosen.exception), 'No runner version was chosen.')
    self.assertEqual(unchosen.exception.fix, '')

  def test_the_development_build_comes_last_and_runs_as_it_is(self):
    """Its commit is unknown until it runs, unless its files tell it."""
    fake_builds.keep(self.kept, fake_builds.OLD)
    runner = self.tmp_path / 'bin/runner'
    runner.parent.mkdir()
    runner.write_text('#!/bin/sh\n')
    builds = self._builds(runner)
    listed = builds.list()
    self.assertEqual([build.id for build in listed], [fake_builds.OLD, _DEV])
    development = listed[1]
    self.assertEqual(development.commit, '')
    self.assertEqual(
        development.label, 'Development build · commit known at start'
    )
    self.assertTrue(development.available)
    runner.write_text('#!/bin/sh\necho changed\n')
    self.assertEqual(builds.runner(_DEV), runner)

  def test_a_kept_copy_as_the_development_build_has_its_commit(self):
    """A runner with build.json beside it: that build.json's commit."""
    directory = fake_builds.keep(self.kept, fake_builds.NEW)
    (development,) = runner_builds.RunnerBuilds(
        None, directory / 'runner'
    ).list()
    self.assertEqual(development.id, _DEV)
    self.assertEqual(development.commit, fake_builds.NEW)
    self.assertEqual(
        development.label, 'Development build · LiteRT-LM 222222222222'
    )

  def test_a_bazel_runner_has_the_commit_of_its_library_repository(self):
    """commit.json beside the library its runfiles name, if not newer."""
    repository = self.tmp_path / 'external/litert_lm_macos_arm64'
    repository.mkdir(parents=True)
    (repository / 'liblitert-lm.dylib').write_bytes(b'library')
    commit_json = repository / 'commit.json'
    commit_json.write_text(
        json.dumps({
            'commit': fake_builds.OLD,
            'commit_date': '2026-10-12T22:51:09-07:00',
            'cl': '998385386',
        })
    )
    solib = self.tmp_path / 'bin/_solib/liblitert-lm.dylib'
    solib.parent.mkdir(parents=True)
    solib.symlink_to(repository / 'liblitert-lm.dylib')
    runner = self.tmp_path / 'bin/cli/runner'
    runner.parent.mkdir()
    runner.write_text('#!/bin/sh\n')
    runner.with_name('runner.runfiles_manifest').write_text(
        f'_main/cli/runner {runner}\n'
        f'_main/_solib/x/liblitert-lm.dylib {solib}\n'
    )
    os.utime(commit_json, (1000, 1000))
    (development,) = runner_builds.RunnerBuilds(None, runner).list()
    self.assertEqual(development.commit, fake_builds.OLD)
    self.assertEqual(development.cl, '998385386')
    # Fetched again after the runner was linked: not known until it runs.
    os.utime(commit_json, (runner.stat().st_mtime + 10,) * 2)
    (development,) = runner_builds.RunnerBuilds(None, runner).list()
    self.assertEqual(development.commit, '')


if __name__ == '__main__':
  test_main.main()
