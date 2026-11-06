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

"""The command-line runner builds this computer can launch.

Two kinds:
- a kept build: a directory `<commit>/` of the kept builds' directory
  (DEFAULT_KEPT), which src/runner's `cli:keep` writes: the runner, the six
  LIBRARIES it loads from beside itself, and build.json, which names the
  LiteRT-LM commit, its date and CL, and the SHA-256 of each file; nothing
  else there may be loaded or run;
- the development build: one runner file, by default the one Bazel built
  last in src/runner; its id is DEVELOPMENT.

Listing reads the directories on every call and launches nothing, so a build
kept while the server runs shows up at once. What a build reports is learned
when it runs: its side then checks the commit (session/side.py).
"""

import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import re
import shutil
from typing import Any

# The id of the development build, as a Session records its choice of it.
DEVELOPMENT = 'dev'
# The id of the PyTorch runner, the one build of PyTorch runs.
PYTORCH = 'pytorch'
# The runtimes a build hosts, as the UI names them.
LITERT_LM = 'LiteRT-LM'
PYTORCH_RUNTIME = 'PyTorch'
# Where `cli:keep` keeps the builds of this computer.
DEFAULT_KEPT = (
    pathlib.Path.home()
    / 'Library/Application Support/model_debugger/runners/macos_arm64'
)
# The libraries a kept build's runner loads from beside itself, as
# src/runner's cli/keep.sh keeps them: LiteRT-LM's library, the two it links
# and its three GPU plugins (llm/litert_lm/litert_lm_library.bzl, macos_arm64).
LIBRARIES = (
    'liblitert-lm.dylib',
    'libLiteRt.dylib',
    'libGemmaModelConstraintProvider.dylib',
    'libwebgpu_dawn.dylib',
    'libLiteRtWebGpuAccelerator.dylib',
    'libLiteRtTopKWebGpuSampler.dylib',
)
_RUNNER = 'runner'
_BUILD_JSON = 'build.json'
# Where a commit with no date it can be ordered by sorts: after every other.
_UNDATED = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
# The library whose LiteRT-LM repository holds the development build's
# commit.json.
_LIBRARY = 'liblitert-lm.dylib'
_COMMIT = re.compile(r'[0-9a-f]{40}')
_CHUNK = 1 << 20


@dataclasses.dataclass(frozen=True)
class Build:
  """A runner build, as read without launching it.

  Attributes:
      id: The LiteRT-LM commit of a kept build; DEVELOPMENT for the development
        build.
      runner: The command-line runner to run.
      commit: The LiteRT-LM commit it was built against; '' when it is known
        only once it runs.
      commit_date: When that commit was made (ISO 8601), or ''.
      cl: The commit's CL, or ''.
      sha256: The SHA-256 of each file of a kept build, by name, as build.json
        has them; empty for the development build, which is run as it is.
      reason: Why it cannot be launched; '' when it can.
      runtime: The runtime its runner hosts: LITERT_LM or PYTORCH_RUNTIME.
      command: The command that runs it, if not its runner alone (the PyTorch
        runner's).
  """

  id: str
  runner: pathlib.Path
  commit: str
  commit_date: str
  cl: str
  sha256: dict[str, str]
  reason: str
  runtime: str = LITERT_LM
  command: tuple[str, ...] = ()

  @property
  def available(self) -> bool:
    """Whether it can be launched, as far as its files tell."""
    return not self.reason

  @property
  def label(self) -> str:
    """What the user knows it by; unique among the builds listed."""
    if self.id == PYTORCH:
      return 'PyTorch runner'
    if self.id == DEVELOPMENT:
      return 'Development build · ' + (
          f'LiteRT-LM {self.commit[:12]}'
          if self.commit
          else 'commit known at start'
      )
    date = self.commit_date[:10]
    return f'LiteRT-LM {self.id[:12]}' + (f' · {date}' if date else '')


class NotThereError(RuntimeError):
  """No build was chosen, or the one chosen is no longer on this computer.

  Its message says what is wrong and how to bring the build back. Choosing
  another version is the other way out, and how depends on whether the
  Session has run (session/side.py).

  Attributes:
      problem: What is wrong, as a clause.
      fix: How to bring the build back, as a clause; '' when none was chosen.
  """

  def __init__(self, problem: str, fix: str) -> None:
    """Holds what is wrong and how to bring the build back.

    Args:
        problem: What is wrong, as a clause.
        fix: How to bring the build back, as a clause, or ''.
    """
    super().__init__(f'{problem}; {fix}.' if fix else f'{problem}.')
    self.problem = problem
    self.fix = fix


class RunnerBuilds:
  """The kept builds of a directory, and a development build."""

  def __init__(
      self,
      kept: pathlib.Path | None,
      development: pathlib.Path | None,
      pytorch: tuple[str, ...] | None = None,
      pytorch_setup: str = '',
  ) -> None:
    """Lists the builds kept in `kept` and the runner `development`.

    Args:
        kept: The kept builds' directory, or None for none; a missing one keeps
          none.
        development: The development build's runner, or None for none; a missing
          file is none.
        pytorch: The command that runs the PyTorch runner, or None for none.
        pytorch_setup: How to make that command, as a clause, for the reason it
          cannot be launched while it is not there.
    """
    self._kept = kept
    self._development = development
    self._pytorch = tuple(pytorch) if pytorch else None
    self._pytorch_setup = pytorch_setup

  @property
  def kept_directory(self) -> pathlib.Path | None:
    """The kept builds' directory, as given."""
    return self._kept

  def list(self) -> list[Build]:
    """Returns every build: kept ones, newest commit first, then dev's."""
    builds = []
    if self._kept is not None and self._kept.is_dir():
      builds = [
          _kept_build(directory)
          for directory in self._kept.iterdir()
          if directory.is_dir() and _COMMIT.fullmatch(directory.name)
      ]
      builds.sort(key=lambda build: build.id)
      builds.sort(
          key=lambda build: _commit_time(build.commit_date),
          reverse=True,
      )
    development = self._development
    if development is not None and development.is_file():
      builds.append(
          Build(
              DEVELOPMENT,
              development,
              sha256={},
              reason='',
              **_development_commit(development),
          )
      )
    if self._pytorch is not None:
      reason = ''
      if shutil.which(self._pytorch[0]) is None:
        reason = f'{self._pytorch[0]} is not there' + (
            f'; {self._pytorch_setup}' if self._pytorch_setup else ''
        )
      builds.append(
          Build(
              PYTORCH,
              pathlib.Path(self._pytorch[0]),
              commit='',
              commit_date='',
              cl='',
              sha256={},
              reason=reason,
              runtime=PYTORCH_RUNTIME,
              command=self._pytorch,
          )
      )
    return builds

  def command(self, build_id: str | None) -> tuple[str, ...]:
    """Returns the command that runs a build, checked as runner() does.

    Args:
      build_id: A build's id, as a Session chose it; None for none.

    Raises:
      NotThereError: No build was chosen, or it is no longer there.
      RuntimeError: It cannot be launched, or a file of it changed.
    """
    if build_id == PYTORCH:
      if self._pytorch is None:
        raise NotThereError(
            'This server runs no PyTorch runner',
            'start it with --pytorch_runner',
        )
      build = next(b for b in self.list() if b.id == PYTORCH)
      if not build.available:
        raise RuntimeError(f'{build.label} cannot be launched: {build.reason}.')
      return self._pytorch
    return (str(self.runner(build_id)),)

  def runner(self, build_id: str | None) -> pathlib.Path:
    """Returns the runner of a build to launch, once its files are checked.

    Each file of a kept build must still have the SHA-256 build.json
    records; the development build is run as it is.

    Args:
        build_id: A build's id, as a Session chose it; None for none.

    Raises:
        NotThereError: No build was chosen, or it is no longer there.
        RuntimeError: It cannot be launched, or a file of it changed since
          it was kept.
    """
    if not build_id:
      raise NotThereError('No runner version was chosen', '')
    build = next((b for b in self.list() if b.id == build_id), None)
    if build is None:
      if build_id == DEVELOPMENT:
        raise NotThereError(
            'The development build of the runner is no longer on this computer',
            'build it again',
        )
      raise NotThereError(
          f'The runner for LiteRT-LM {build_id[:12]} is no longer kept '
          'on this computer',
          'keep it again',
      )
    if not build.available:
      raise RuntimeError(f'{build.label} cannot be launched: {build.reason}')
    for name, expected in build.sha256.items():
      if _sha256(build.runner.parent / name) != expected:
        raise RuntimeError(
            f'{build.label} was changed after it was kept: {name} does '
            'not match the SHA-256 in its build.json. Keep it again.'
        )
    return build.runner


def _kept_build(directory: pathlib.Path) -> Build:
  """Reads a kept build's directory; launches nothing, hashes nothing."""
  commit = directory.name
  runner = directory / _RUNNER

  def unavailable(reason: str, facts: dict[str, Any] | None = None) -> Build:
    facts = facts or {}
    return Build(
        commit,
        runner,
        commit,
        _string(facts.get('commit_date')),
        _string(facts.get('cl')),
        {},
        reason,
    )

  try:
    facts = _read_json(directory / _BUILD_JSON)
  except (OSError, ValueError) as error:
    return unavailable(
        f'{_BUILD_JSON} cannot be read ({error}). Keep it again.'
    )
  sha256 = facts.get('sha256')
  if (
      not isinstance(sha256, dict)
      or not all(
          isinstance(name, str) and isinstance(value, str)
          for name, value in sha256.items()
      )
      or any('/' in name or name.startswith('.') for name in sha256)
  ):
    return unavailable(
        f'{_BUILD_JSON} lists no SHA-256 of its files. Keep it again.',
        facts,
    )
  if facts.get('commit') != commit:
    return unavailable(
        f'{_BUILD_JSON} names LiteRT-LM {_string(facts.get("commit"))!r}, '
        'not the commit of its directory. Keep it again.',
        facts,
    )
  for name in (_RUNNER, *LIBRARIES):
    if name not in sha256:
      what = 'the runner' if name == _RUNNER else name
      return unavailable(
          f'{_BUILD_JSON} lists no SHA-256 of {what}. Keep it again.',
          facts,
      )
  for name in sorted(sha256):
    if not (directory / name).is_file():
      return unavailable(f'{name} is missing. Keep it again.', facts)
  # A library or program it was not kept with could be loaded or run
  # unchecked.
  for path in sorted(directory.iterdir()):
    if path.name not in sha256 and _loadable(path):
      return unavailable(
          f'{path.name} is not one of the files it was kept with. '
          'Keep it again.',
          facts,
      )
  return Build(
      commit,
      runner,
      commit,
      _string(facts.get('commit_date')),
      _string(facts.get('cl')),
      dict(sha256),
      '',
  )


def _loadable(path: pathlib.Path) -> bool:
  """Whether a file of a kept build's directory is a library or program."""
  try:
    return path.is_file() and (
        path.suffix == '.dylib' or bool(path.stat().st_mode & 0o111)
    )
  except OSError:
    return True


def _commit_time(commit_date: str) -> datetime.datetime:
  """When a commit was made, from its ISO 8601 date with an offset.

  A date that cannot be read, or has no offset, cannot be ordered against
  the others: _UNDATED.

  Args:
    commit_date: An ISO 8601 date string.

  Returns:
    The parsed datetime with timezone, or _UNDATED if invalid/offset-less.
  """
  try:
    time = datetime.datetime.fromisoformat(commit_date)
  except ValueError:
    return _UNDATED
  return time if time.tzinfo is not None else _UNDATED


def _development_commit(runner: pathlib.Path) -> dict[str, Any]:
  """The development build's commit, date and CL, if known without a run.

  A kept build's copy has its build.json beside it. A runner Bazel built
  loads liblitert-lm.dylib from the LiteRT-LM repository its runfiles
  manifest names, whose commit.json names the commit; Bazel fetches that
  repository again when the commit changes, so a commit.json newer than
  the runner may not be the one it was linked against, and is not used.

  Args:
    runner: The path to the runner executable.

  Returns:
    `commit`, `commit_date` and `cl`, each '' when not known.
  """
  facts = {}
  try:
    beside = runner.parent / _BUILD_JSON
    if beside.is_file():
      facts = _read_json(beside)
    else:
      commit_json = _repository_commit_json(runner)
      if (
          commit_json is not None
          and commit_json.stat().st_mtime <= runner.stat().st_mtime
      ):
        facts = _read_json(commit_json)
  except (OSError, ValueError):
    facts = {}
  commit = _string(facts.get('commit'))
  if not _COMMIT.fullmatch(commit):
    return {'commit': '', 'commit_date': '', 'cl': ''}
  return {
      'commit': commit,
      'commit_date': _string(facts.get('commit_date')),
      'cl': _string(facts.get('cl')),
  }


def _repository_commit_json(runner: pathlib.Path) -> pathlib.Path | None:
  """The commit.json beside the LiteRT-LM library a Bazel runner loads."""
  manifest = runner.with_name(runner.name + '.runfiles_manifest')
  if not manifest.is_file():
    return None
  for line in manifest.read_text(encoding='utf-8').splitlines():
    name, _, target = line.partition(' ')
    if name.endswith('/' + _LIBRARY) and target:
      return pathlib.Path(os.path.realpath(target)).parent / 'commit.json'
  return None


def _read_json(path: pathlib.Path) -> dict[str, Any]:
  """The JSON object of a file; ValueError if it holds something else."""
  with open(path, encoding='utf-8') as file:
    value = json.load(file)
  if not isinstance(value, dict):
    raise ValueError('not a JSON object')
  return value


def _string(value: object) -> str:
  return value if isinstance(value, str) else ''


def _sha256(path: pathlib.Path) -> str:
  """The SHA-256 of a file; '' when it cannot be read."""
  digest = hashlib.sha256()
  try:
    with open(path, 'rb') as file:
      while chunk := file.read(_CHUNK):
        digest.update(chunk)
  except OSError:
    return ''
  return digest.hexdigest()
