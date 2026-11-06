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

"""Models named on Hugging Face: resolved, downloaded and recorded.

The editor names a side's model by a Hugging Face link or repository id
(parse()): for LiteRT-LM one `.litertlm` file of a repository, for PyTorch a
whole repository, at a revision ('main' unless named). Before it is chosen,
the editor says whether the server has it already, and how large it is
(check()). A download job (jobs, operation 'download') resolves it with
huggingface_hub, downloads what the server does not have yet into the
workspace's own Hugging Face cache, `hf/` (workspace), where it stays as
long as a Session uses it, and records it in the model store
(model_store.ModelStore.add_download()): a LiteRT-LM file by its SHA-256, a
PyTorch repository's snapshot directory by its manifest, so that the same
files are the same model whichever way they came. The model keeps where it
came from (Model.source): the repository, the file, the revision named and
the commit it resolved to. Nothing is read from or written to the shared
Hugging Face cache of the server's computer, the Xet cache included.

Of a PyTorch repository only what the PyTorch runner reads is fetched
(PYTORCH_PATTERNS): its configs, tokenizer, chat template and safetensors;
one without a config.json is refused. The token, for gated repositories
such as Gemma's, is huggingface_hub's own: HF_TOKEN, or the one `hf auth
login` saved; it is never read here, nor shown. A file the server has is not
fetched again: a revision that is a commit is resolved from its cache alone,
a branch with one request.

A download runs in a child process of its own (_ChildTransfer), so that it
can be stopped at once: huggingface_hub, and the Xet transfers of hf_xet
under it, take no request to stop, and a thread cannot be killed. The child
keeps hf_xet's cache in the workspace (`hf/xet`), shows no progress bar, and
counts the bytes of huggingface_hub's own byte bars, which the job's
progress says (the bytes received of a Xet file, before it is
reconstructed on disk).

A LiteRT-LM file is one model by its content: before one is downloaded,
the hub is asked its SHA-256 (its LFS object id), and a model of that
content the server keeps already, an upload say, is used instead, with no
download (held()). Without the hub's answer, offline say, the file is
fetched as ever.

A download may be cancelled (cancel()): its job ends at once, cancelled,
its child is stopped, and what it wrote is deleted then (_remove_partial()):
each incomplete file, under huggingface_hub's own lock of it unless another
download holds it, each file it finished that no model holds, and each link
left to nothing, while the repository is still marked as being written in
(model_store.ModelStore.downloading()).
"""

from collections.abc import Callable
import dataclasses
import importlib
import logging as std_logging
import multiprocessing
import os
import pathlib
import re
import threading
from typing import Any
import urllib.parse

from absl import logging
import huggingface_hub
from huggingface_hub import errors as hub_errors
from huggingface_hub import utils as hub_utils
from model_debugger.server.session import jobs as jobs_lib
from model_debugger.server.session import model_store

HOST = 'huggingface.co'
# What a PyTorch repository is fetched of: what its runner reads.
PYTORCH_PATTERNS = (
    '*.json',
    '*.safetensors',
    '*.jinja',
    'tokenizer*',
    '*.model',
    '*.txt',
)
# How often a download's progress is read.
_POLL_SECONDS = 0.5
_REPO = re.compile(r'[A-Za-z0-9][\w.-]*/[\w.-]+')
_REVISION = re.compile(r'[\w./-]{1,128}')
# What the editor's Hugging Face form says, by runtime.
_FORMS = {
    model_store.LITERT_LM: {
        'link_label': 'Hugging Face link or repository',
        'link_placeholder': (
            'https://huggingface.co/org/repo/blob/main/model.litertlm or '
            'org/repo'
        ),
        'file_label': 'LiteRT-LM file',
        'file_placeholder': 'model.litertlm (if the repository has several)',
        'revision_label': 'Revision',
        'revision_placeholder': 'main',
        'help': (
            'One .litertlm file of a Hugging Face repository, downloaded to '
            'the Server unless it is there already. A gated repository needs '
            "the Server's Hugging Face token (HF_TOKEN, or `hf auth login`)."
        ),
        'action': 'Use model',
    },
    model_store.PYTORCH: {
        'link_label': 'Hugging Face link or repository',
        'link_placeholder': 'https://huggingface.co/org/repo or org/repo',
        'file_label': '',
        'file_placeholder': '',
        'revision_label': 'Revision',
        'revision_placeholder': 'main',
        'help': (
            'A Hugging Face model repository: its configs, tokenizer and '
            'safetensors are downloaded to the Server unless they are there '
            "already. A gated repository needs the Server's Hugging Face "
            'token (HF_TOKEN, or `hf auth login`).'
        ),
        'action': 'Use model',
    },
}


@dataclasses.dataclass(frozen=True)
class Request:
  """A model named on Hugging Face, checked.

  Attributes:
      runtime: model_store.LITERT_LM or model_store.PYTORCH.
      repo: The repository, `<owner>/<name>`.
      file: The `.litertlm` file's path in it; '' for a PyTorch model, or for a
        LiteRT-LM one whose repository has one `.litertlm` file.
      revision: The branch, tag or commit; 'main' unless named.
  """

  runtime: str
  repo: str
  file: str
  revision: str


def form(runtime: str) -> dict[str, Any]:
  """What the editor's Hugging Face form of a runtime says."""
  return dict(_FORMS[runtime])


def parse(
    runtime: Any, link: Any, file: Any = '', revision: Any = ''
) -> Request:
  """Checks what the editor names; returns the request.

  Args:
    runtime: The side's runtime.
    link: A link of huggingface.co (a repository, `/tree/<revision>`, or a
      file's `/blob/` or `/resolve/<revision>/<path>`), or a repository id
      `<owner>/<name>`.
    file: The file's path, for a LiteRT-LM repository id; a link to a file names
      its own.
    revision: The revision, if the link names none; '' for 'main'.

  Returns:
    The parsed Request object.

  Raises:
    ValueError: Any of them is not valid, a LiteRT-LM file is not a `.litertlm`
      one, or a file is named for a PyTorch model.
  """
  if runtime not in _FORMS:
    raise ValueError('Choose LiteRT-LM or PyTorch.')
  if not all(isinstance(v, str) for v in (link, file, revision)):
    raise ValueError('Name the model by a Hugging Face link or repository.')
  link, file, revision = link.strip(), file.strip(), revision.strip()
  repo = link
  if '://' in link:
    url = urllib.parse.urlparse(link)
    if url.scheme != 'https' or url.hostname != HOST:
      raise ValueError(f'A link must be one of https://{HOST}/.')
    parts = [urllib.parse.unquote(p) for p in url.path.split('/') if p]
    if len(parts) < 2:
      raise ValueError(f'Name a repository: https://{HOST}/<owner>/<name>.')
    repo = '/'.join(parts[:2])
    rest = parts[2:]
    if rest and rest[0] in ('blob', 'resolve') and len(rest) >= 3:
      revision, file = rest[1], '/'.join(rest[2:])
    elif rest and rest[0] == 'tree' and len(rest) >= 2:
      revision = rest[1]
    elif rest:
      raise ValueError(
          'Link to the repository, or to a file of it (/blob/<revision>/'
          '<file>).'
      )
  if not _REPO.fullmatch(repo) or '..' in repo:
    raise ValueError(
        f'{repo or "The repository"} is not a Hugging Face repository: '
        'name it <owner>/<name>.'
    )
  revision = revision or 'main'
  if not _REVISION.fullmatch(revision) or '..' in revision.split('/'):
    raise ValueError(f'{revision} is not a revision.')
  if file and (
      file.startswith('/')
      or '\\' in file
      or '..' in file.split('/')
      or any(ord(char) < 32 for char in file)
  ):
    raise ValueError(f'{file} is not a file of a repository.')
  if runtime == model_store.PYTORCH:
    if file:
      raise ValueError(
          'A PyTorch model is a whole repository: link to the '
          'repository, not to one of its files.'
      )
  elif file and not file.endswith(model_store.MODEL_SUFFIX):
    raise ValueError(
        f'{file} is not a LiteRT-LM model: choose a '
        f'{model_store.MODEL_SUFFIX} file.'
    )
  return Request(runtime, repo, file, revision)


def failure(error: BaseException, request: Request) -> str:
  """What a failed resolve or download says, as the editor shows it."""
  where = request.repo + (f'/{request.file}' if request.file else '')
  if isinstance(error, hub_errors.GatedRepoError):
    return (
        f'{request.repo} is gated: accept its terms on {HOST} with your '
        'account, and give this server your token (HF_TOKEN, or `hf '
        'auth login`).'
    )
  if isinstance(error, hub_errors.RevisionNotFoundError):
    return f'{request.repo} has no revision {request.revision}.'
  if isinstance(error, hub_errors.RemoteEntryNotFoundError):
    return f'{request.repo} has no file {request.file} at {request.revision}.'
  if isinstance(error, hub_errors.RepositoryNotFoundError):
    return (
        f'{request.repo} is not a repository this server can read: it '
        'does not exist, or it is private and the token (HF_TOKEN, or '
        '`hf auth login`) does not give access to it.'
    )
  if isinstance(error, hub_errors.LocalEntryNotFoundError) or isinstance(
      error, (hub_errors.OfflineModeIsEnabled, ConnectionError, TimeoutError)
  ):
    return f'{where} is not on the Server, and {HOST} cannot be reached.'
  if isinstance(error, hub_errors.HfHubHTTPError):
    status = getattr(error.response, 'status_code', None)
    if status in (401, 403):
      return (
          f'{HOST} refused {where} ({status}): the token (HF_TOKEN, or '
          '`hf auth login`) does not give access to it.'
      )
    return f'{HOST} could not give {where}: {_first_line(error)}'
  if isinstance(error, ValueError):
    return _first_line(error)
  if isinstance(error, OSError):
    return f'{where}: {_first_line(error)}'
  return f'{where} could not be fetched: {_first_line(error)}'


def _first_line(error: BaseException) -> str:
  return (str(error).strip().splitlines() or [type(error).__name__])[0]


def size_text(size: int) -> str:
  """Bytes as the editor shows them, in GB or MB."""
  if size >= 10**9:
    return f'{size / 10**9:.1f} GB'
  return f'{size / 10**6:.1f} MB'


# What check() says of a model the server has, and of one it would download.
ON_SERVER = 'On the Server'
DOWNLOADS = 'Downloads to the Server'
# Why a download job ended cancelled.
CANCELLED = 'The download was cancelled.'
# How long a cancelled download's clean-up waits for another's lock.
_LOCK_SECONDS = 1.0


class _CancelledError(Exception):
  """A download was cancelled (HfModels.cancel())."""


class _StoppedError(Exception):
  """A download in a thread was stopped (_ThreadTransfer.stop())."""


def _byte_bar(
    record: Callable[[float], None], stopped: Callable[[], bool]
) -> type[Any]:
  """A tqdm class for huggingface_hub that counts what its byte bars say.

  Each bar of bytes (unit 'B') counts what it was updated by, and `record`
  is given its count each time; a bar of files counts nothing. Once
  `stopped()`, an update raises _StoppedError.

  Args:
    record: Callback receiving updated byte counts.
    stopped: Predicate returning whether transfer is stopped.

  Returns:
    A subclass of tqdm that reports downloaded byte counts.
  """

  class ByteBar(hub_utils.tqdm):
    """Custom tqdm class to record downloaded bytes and check cancellation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
      self._counts = kwargs.get('unit') == 'B'
      self._done = 0.0
      super().__init__(*args, **kwargs)

    def update(self, n: float | None = 1) -> bool | None:
      if stopped():
        raise _StoppedError()
      if self._counts and n:
        self._done += n
        record(self._done)
      return super().update(n)

  return ByteBar


def _child(
    target: str,
    args: list[Any],
    kwargs: dict[str, Any],
    env: dict[str, str],
    counter: Any,
    send: Any,
) -> None:
  """A download's child process: runs `target` and sends what it returned.

  Args:
      target: `<module>:<function>`, such as 'huggingface_hub:hf_hub_download'.
      args: Its positional arguments.
      kwargs: Its keyword arguments; `_progress` True gives it a `tqdm_class`
        that counts its bytes in `counter`.
      env: Set in the environment first, before huggingface_hub reads it.
      counter: A shared integer: the most bytes a byte bar counted.
      send: Where `(True, result)` or `(False, error)` goes.
  """
  os.environ.update(env)

  def record(done: float) -> None:
    with counter.get_lock():
      if done > counter.value:
        counter.value = int(done)

  kwargs = dict(kwargs)
  if kwargs.pop('_progress', False):
    kwargs['tqdm_class'] = _byte_bar(record, lambda: False)
  module, name = target.split(':')
  try:
    function = getattr(importlib.import_module(module), name)
    outcome: tuple[bool, Any] = (True, function(*args, **kwargs))
  except BaseException as error:  # pylint: disable=broad-except
    try:
      send.send((False, error))
    except (TypeError, AttributeError, ValueError, OSError):
      send.send((False, RuntimeError(f'{type(error).__name__}: {error}')))
    send.close()
    return
  send.send(outcome)
  send.close()


class _ChildTransfer:
  """A download in a child process, which stop() ends at once."""

  def __init__(
      self,
      target: str,
      args: list[Any],
      kwargs: dict[str, Any],
      env: dict[str, str],
  ) -> None:
    """Starts `target(*args, **kwargs)` in a child process (_child())."""
    context = multiprocessing.get_context('spawn')
    self._counter = context.Value('q', 0)
    self._receive, send = context.Pipe(duplex=False)
    self._process = context.Process(
        target=_child,
        args=(target, args, kwargs, env, self._counter, send),
        name='hf-download',
        daemon=True,
    )
    self._process.start()
    send.close()
    self._outcome: tuple[bool, Any] | None = None

  def done(self) -> int:
    """The bytes it fetched so far, as its byte bars say."""
    return self._counter.value

  def alive(self) -> bool:
    return self._outcome is None and self._process.is_alive()

  def wait(self, seconds: float) -> None:
    """Waits up to `seconds` for it to end."""
    if self._receive.poll(seconds):
      self._take()

  def stop(self) -> None:
    """Ends the child now; returns once it is gone."""
    self._process.terminate()
    self._process.join(5)
    if self._process.is_alive():
      self._process.kill()
      self._process.join()

  def result(self) -> Any:
    """What it returned, once it ended.

    Returns:
      The value returned by the child process target function.

    Raises:
      Exception: What it raised, or that it ended without a word.
    """
    while self._outcome is None and self._process.is_alive():
      self.wait(0.2)
    self._take()
    self._process.join()
    if self._outcome is None:
      raise RuntimeError(
          f'The download stopped (exit code {self._process.exitcode}).'
      )
    ok, value = self._outcome
    if not ok:
      raise value
    return value

  def _take(self) -> None:
    if self._outcome is not None:
      return
    try:
      if self._receive.poll():
        self._outcome = self._receive.recv()
    except (EOFError, OSError):
      # It ended without a word: it is gone, or about to be.
      self._process.join(1)
    except Exception as error:  # pylint: disable=broad-except
      self._outcome = (False, RuntimeError(_errors_text(error)))


def _errors_text(error: BaseException) -> str:
  """What a download's error that could not be read back says."""
  return f'The download failed: {type(error).__name__}: {error}'


class _ThreadTransfer:
  """A download in a thread, of a hub given in its place (tests).

  It stops at its next byte once `stopped` is set (_byte_bar()); stop()
  returns once it has.
  """

  def __init__(
      self,
      function: Callable[..., Any],
      kwargs: dict[str, Any],
      stopped: threading.Event,
  ) -> None:
    self._stopped = stopped
    self._lock = threading.Lock()
    self._done = 0
    self._found: dict[str, Any] = {}
    bar = _byte_bar(self._record, self._stopped.is_set)
    self._thread = threading.Thread(
        target=self._run,
        args=(function, {**kwargs, 'tqdm_class': bar}),
        name='hf-fetch',
        daemon=True,
    )
    self._thread.start()

  def _record(self, done: float) -> None:
    with self._lock:
      self._done = max(self._done, int(done))

  def _run(self, function: Callable[..., Any], kwargs: dict[str, Any]) -> None:
    try:
      self._found['result'] = function(**kwargs)
    except BaseException as error:  # pylint: disable=broad-except
      self._found['error'] = error

  def done(self) -> int:
    with self._lock:
      return self._done

  def alive(self) -> bool:
    return self._thread.is_alive()

  def wait(self, seconds: float) -> None:
    self._thread.join(seconds)

  def stop(self) -> None:
    self._stopped.set()
    self._thread.join()

  def result(self) -> Any:
    self._thread.join()
    if 'error' in self._found:
      raise self._found['error']
    return self._found['result']


class HfModels:
  """Fetches the models the editor names on Hugging Face, one job each."""

  def __init__(
      self,
      models: model_store.ModelStore,
      jobs: jobs_lib.Jobs,
      cache: pathlib.Path,
      hub: Any = None,
  ) -> None:
    """Records what it fetches in `models`; reports in `jobs`.

    Args:
        models: Where the models are recorded.
        jobs: Where each download is a job.
        cache: The workspace's Hugging Face cache (workspace.hf), where the
          downloads go.
        hub: What has huggingface_hub's hf_hub_download, snapshot_download,
          list_repo_files and get_paths_info, its downloads run in a thread
          (_ThreadTransfer); None for huggingface_hub, whose downloads run in a
          child process each (_ChildTransfer), their Xet cache in `cache` too.
    """
    self._models = models
    self._jobs = jobs
    self._cache = cache
    self._spawn = hub is None
    self._hub = hub or huggingface_hub
    if self._spawn:
      # The server's log is no terminal: huggingface_hub's bars (of its
      # dry runs here) and each request its client makes stay out.
      hub_utils.disable_progress_bars()
      std_logging.getLogger('httpx').setLevel(std_logging.WARNING)
    # Guards _stops and the end of each download's job.
    self._lock = threading.Lock()
    # Each running download's job id -> what cancel() sets.
    self._stops: dict[str, threading.Event] = {}

  def check(self, payload: Any) -> dict[str, Any]:
    """Says whether the server has what the editor names, and its size.

    Asks the hub what a download would fetch, as the download does;
    without an answer (offline, say), the model store says whether a
    model of this repository, file and revision is on the server. A
    LiteRT-LM file the server keeps the content of is on the server
    too, whatever way it came (held()).

    Args:
        payload: `{runtime, link, file, revision}` (parse()).

    Returns:
        `{on_server, size, text}`: whether the server has it; its bytes,
        or those to download, None if unknown; and the editor's line,
        ON_SERVER or DOWNLOADS, then the size if known.

    Raises:
        ValueError: What it names is not valid (parse()), or the hub
          refuses it (failure() says why).
    """
    request = self._request(payload)
    try:
      plan = self._check_plan(request)
    except Exception as error:  # pylint: disable=broad-except
      raise ValueError(failure(error, request)) from None
    if plan is None:
      known = next(
          (
              model
              for model in self._models.list()
              if model.present()
              and model.runtime == request.runtime
              and model.source
              and model.source.get('repo') == request.repo
              and model.source.get('revision') == request.revision
              and (not request.file or model.source.get('file') == request.file)
          ),
          None,
      )
      on_server = known is not None
      size = known.size if known is not None else None
    else:
      fetching = [info for info in plan if info.will_download]
      on_server = not fetching
      size = sum(info.file_size or 0 for info in fetching or plan) or None
      if fetching and request.runtime == model_store.LITERT_LM:
        held = self.held(request)
        if held is not None:
          on_server, size = True, held.size
    text = ON_SERVER if on_server else DOWNLOADS
    if size:
      text += f' · {size_text(size)}'
    return {'on_server': on_server, 'size': size, 'text': text}

  def _check_plan(self, request: Request) -> list[Any] | None:
    """What a download of `request` would fetch; None if unknown."""
    if request.runtime == model_store.PYTORCH:
      return self._plan(
          lambda: self._hub.snapshot_download(
              **self._repository_names(request), dry_run=True
          )
      )
    try:
      file = request.file or self._only_model_file(request)
    except (
        hub_errors.OfflineModeIsEnabled,
        ConnectionError,
        TimeoutError,
    ):
      return None
    return self._plan(
        lambda: [
            self._hub.hf_hub_download(
                **self._file_names(request, file), dry_run=True
            )
        ]
    )

  def held(self, request: Request, file: str = '') -> model_store.Model | None:
    """The model the server keeps of a LiteRT-LM file's content, if any.

    The hub is asked the file's SHA-256 (its LFS object id); without an
    answer (offline, a file not in LFS, a hub that refuses) there is
    none, and the file is fetched as ever.

    Args:
      request: The file named.
      file: Its path in the repository; request.file, or the one `.litertlm`
        file of the repository, if ''.

    Returns:
      The held model if found, or None.
    """
    try:
      file = file or request.file or self._only_model_file(request)
      (info,) = self._hub.get_paths_info(
          request.repo, [file], revision=request.revision
      )
      sha256 = info.lfs.sha256 if info.lfs is not None else ''
    except Exception:  # pylint: disable=broad-except
      return None
    return next(
        (
            model
            for model in self._models.list()
            if sha256
            and model.sha256 == sha256
            and model.runtime == model_store.LITERT_LM
            and model.present()
        ),
        None,
    )

  def _request(self, payload: Any) -> Request:
    """The request `payload` names (parse())."""
    if not isinstance(payload, dict):
      raise ValueError('Name the model by a Hugging Face link.')
    return parse(
        payload.get('runtime'),
        payload.get('link', ''),
        payload.get('file', ''),
        payload.get('revision', ''),
    )

  def _file_names(self, request: Request, file: str) -> dict[str, Any]:
    """hf_hub_download's arguments for one file of `request`."""
    return {
        'repo_id': request.repo,
        'filename': file,
        'revision': request.revision,
        'cache_dir': self._cache,
    }

  def _repository_names(self, request: Request) -> dict[str, Any]:
    """snapshot_download's arguments for `request`'s repository."""
    return {
        'repo_id': request.repo,
        'revision': request.revision,
        'allow_patterns': list(PYTORCH_PATTERNS),
        'cache_dir': self._cache,
    }

  def start(self, payload: Any) -> dict[str, Any]:
    """Starts the download job of what the editor names.

    Args:
        payload: `{runtime, link, file, revision}` (parse()).

    Returns:
        The job: its `model`, once it completed, is `{artifact, name,
        runtime, source}`.

    Raises:
        ValueError: What it names is not valid (parse()).
    """
    request = self._request(payload)
    job = self._jobs.create('', 'download')
    stop = threading.Event()
    with self._lock:
      self._stops[job['id']] = stop
    threading.Thread(
        target=self._run,
        args=(job['id'], request, stop),
        name='hf-download',
        daemon=True,
    ).start()
    return job

  def cancel(self, job_id: str) -> dict[str, Any]:
    """Cancels a download: its job ends at once, cancelled.

    Its child is stopped, and what it wrote is deleted, by the job's own
    thread (_download()). A download that ended already is answered as
    it is.

    Args:
      job_id: The job ID to cancel.

    Returns:
      The job.

    Raises:
      NotFound: No such job.
    """
    with self._lock:
      stop = self._stops.pop(job_id, None)
      if stop is not None:
        stop.set()
        self._jobs.finish(job_id, 'cancelled', CANCELLED)
    if stop is not None:
      logging.info('Download %s cancelled', job_id)
    return self._jobs.get(job_id)

  def fetch(
      self,
      request: Request,
      report: Callable[[str], None],
      stop: threading.Event | None = None,
  ) -> model_store.Model:
    """Resolves, fetches and records a model; what a job runs.

    Args:
      request: What to fetch.
      report: Takes each progress line.
      stop: Set to cancel the download; None never cancels it.

    Returns:
      The model recorded.

    Raises:
      ValueError: The repository has no such model.
      Exception: As huggingface_hub raises (failure() says what).
    """
    report(f'Resolving {request.repo}')
    stop = stop or threading.Event()
    repository = 'models--' + request.repo.replace('/', '--')
    with self._models.downloading(repository):
      if request.runtime == model_store.PYTORCH:
        return self._fetch_repository(request, report, stop)
      return self._fetch_file(request, report, stop)

  def _fetch_file(
      self,
      request: Request,
      report: Callable[[str], None],
      stop: threading.Event,
  ) -> model_store.Model:
    """Fetches a single LiteRT-LM model file from Hugging Face."""
    file = request.file or self._only_model_file(request)
    request = dataclasses.replace(request, file=file)
    names = self._file_names(request, file)
    plan = self._plan(
        lambda: [self._hub.hf_hub_download(**names, dry_run=True)]
    )
    if plan is None or any(info.will_download for info in plan):
      held = self.held(request, file)
      if held is not None:
        report(f'{file} is on the Server: {held.name}')
        return held
    path = self._download(request, plan, 'hf_hub_download', names, report, stop)
    source = _source(request, path, plan)
    return self._models.add_download(pathlib.Path(path), source)

  def _only_model_file(self, request: Request) -> str:
    """The one `.litertlm` file of a repository named without a file."""
    files = [
        name
        for name in self._hub.list_repo_files(
            request.repo, revision=request.revision
        )
        if name.endswith(model_store.MODEL_SUFFIX)
    ]
    if len(files) == 1:
      return files[0]
    if not files:
      raise ValueError(
          f'{request.repo} has no {model_store.MODEL_SUFFIX} file at '
          f'{request.revision}.'
      )
    shown = ', '.join(sorted(files)[:5]) + (', …' if len(files) > 5 else '')
    raise ValueError(
        f'{request.repo} has {len(files)} {model_store.MODEL_SUFFIX} '
        f'files ({shown}): name the one to use.'
    )

  def _fetch_repository(
      self,
      request: Request,
      report: Callable[[str], None],
      stop: threading.Event,
  ) -> model_store.Model:
    """Fetches a full PyTorch model repository snapshot from Hugging Face."""
    names = self._repository_names(request)
    plan = self._plan(
        lambda: self._hub.snapshot_download(**names, dry_run=True)
    )
    if plan is not None and not any(
        info.filename == 'config.json' for info in plan
    ):
      raise ValueError(
          f'{request.repo} has no config.json at {request.revision}: '
          'it is not a Hugging Face model repository the PyTorch '
          'runner reads.'
      )
    path = self._download(
        request, plan, 'snapshot_download', names, report, stop
    )
    if not (pathlib.Path(path) / 'config.json').is_file():
      raise ValueError(
          f'{request.repo} has no config.json at {request.revision}.'
      )
    source = _source(request, path, plan)
    report(f'Reading {request.repo}')
    return self._models.add_download_directory(pathlib.Path(path), source)

  def _plan(self, dry_run: Callable[[], list[Any]]) -> list[Any] | None:
    """What a download would fetch; None if the hub cannot say.

    Without the network, or offline, the download itself then finds what
    the cache holds, or fails.

    Args:
      dry_run: Callable running dry run and returning list of items to fetch.

    Returns:
      List of plan items to download, or None if hub cannot say.
    """
    try:
      return list(dry_run())
    except (
        hub_errors.DryRunError,
        hub_errors.OfflineModeIsEnabled,
        ConnectionError,
        TimeoutError,
    ):
      return None

  def _download(
      self,
      request: Request,
      plan: list[Any] | None,
      kind: str,
      names: dict[str, Any],
      report: Callable[[str], None],
      stop: threading.Event,
  ) -> str:
    """Runs a download of the hub's `kind`, reporting the bytes fetched.

    Once `stop` is set, the download is stopped at once, what it wrote
    is deleted (_remove_partial()), and _CancelledError is raised.

    Args:
      request: What it fetches.
      plan: What the hub said it fetches; None if unknown.
      kind: 'hf_hub_download' or 'snapshot_download'.
      names: Its arguments.
      report: Takes each progress line.
      stop: Set to cancel it.

    Returns:
      Path of the downloaded file or directory.

    Raises:
      _CancelledError: Download was cancelled.
    """
    fetching = [info for info in plan or [] if info.will_download]
    total = sum(info.file_size or 0 for info in fetching)
    cached = plan is not None and not fetching
    if cached:
      report(f'{request.repo} is on the Server')
    transfer = self._transfer(kind, names, stop)
    what = request.file or request.repo
    while transfer.alive():
      if stop.is_set():
        break
      if not cached:
        done = transfer.done()
        if total:
          done = min(done, total)
          report(
              f'Downloading {what} · {size_text(done)} of '
              f'{size_text(total)} · {done * 100 // total}%'
          )
        elif done:
          report(f'Downloading {what} · {size_text(done)}')
        else:
          report(f'Downloading {what}')
      transfer.wait(_POLL_SECONDS)
    if stop.is_set():
      transfer.stop()
      self._remove_partial(request, fetching)
      raise _CancelledError()
    path = transfer.result()
    if not cached:
      report(f'Downloaded {what}' + (f' · {size_text(total)}' if total else ''))
    return path

  def _transfer(
      self, kind: str, names: dict[str, Any], stop: threading.Event
  ) -> Any:
    """Starts a download: in a child process, or the hub's in a thread."""
    if not self._spawn:
      return _ThreadTransfer(getattr(self._hub, kind), names, stop)
    return _ChildTransfer(
        f'huggingface_hub:{kind}',
        [],
        {**names, '_progress': True},
        {
            'HF_XET_CACHE': str(self._cache / 'xet'),
            'HF_HUB_DISABLE_PROGRESS_BARS': '1',
        },
    )

  def _remove_partial(self, request: Request, fetching: list[Any]) -> None:
    """Deletes what a stopped download wrote in its repository's cache.

    Each incomplete blob, under huggingface_hub's lock of it (one
    another download holds is left to it); each file of `fetching` it
    finished that no model holds, with the links to it (a snapshot's
    to the repository's blob, the blob's to the shared store's file);
    and each link of the repository left to nothing.

    Args:
      request: Request being removed.
      fetching: List of file info objects that were planned for fetching.
    """
    repository = 'models--' + request.repo.replace('/', '--')
    folder = self._cache / repository
    held = self._held_files()
    # An incomplete blob is `<etag>[.<suffix>].incomplete`, locked as
    # `<etag>.lock`.
    doomed: list[tuple[str, list[pathlib.Path]]] = [
        (path.name.split('.')[0], [path])
        for path in (folder / 'blobs').glob('*.incomplete')
    ]
    for info in fetching:
      chain = _links(pathlib.Path(info.local_path))
      if not chain[-1].is_file() or chain[-1].resolve() in held:
        continue
      blob = next(
          (path for path in chain if path.parent == folder / 'blobs'),
          chain[-1],
      )
      doomed.append((blob.name, chain))
    for etag, paths in doomed:
      lock = self._cache / '.locks' / repository / f'{etag}.lock'
      lock.parent.mkdir(parents=True, exist_ok=True)
      try:
        with hub_utils.WeakFileLock(lock, timeout=_LOCK_SECONDS):
          for path in paths:
            path.unlink(missing_ok=True)
      except Exception:  # pylint: disable=broad-except
        continue
    for path in folder.rglob('*') if folder.is_dir() else []:
      if path.is_symlink() and not path.exists():
        path.unlink(missing_ok=True)

  def _held_files(self) -> set[pathlib.Path]:
    """Every file a model holds, links resolved: a directory's each."""
    held = set()
    for model in self._models.list():
      if model.path is None:
        continue
      if model.path.is_dir():
        held.update(
            path.resolve() for path in model.path.rglob('*') if path.is_file()
        )
      else:
        held.add(model.path.resolve())
    return held

  def _run(self, job_id: str, request: Request, stop: threading.Event) -> None:
    """Worker running in a background thread to fetch a model and report progress."""

    def report(message: str) -> None:
      if not stop.is_set():
        self._jobs.emit(job_id, 'progress', message=message)

    try:
      model = self.fetch(request, report, stop)
    except _CancelledError:
      return
    except Exception as error:  # pylint: disable=broad-except
      if stop.is_set():
        logging.warning(
            'Download %s, cancelled, could not be cleared: %r',
            job_id,
            error,
        )
      self._end(job_id, 'failed', failure(error, request))
      return
    with self._lock:
      if stop.is_set():
        return
      self._jobs.emit(
          job_id,
          'model',
          model={
              'artifact': model.artifact,
              'name': model.name,
              'runtime': model.runtime,
              'source': model.source,
          },
      )
    self._end(job_id, 'completed')

  def _end(self, job_id: str, status: str, error: str = '') -> None:
    """Ends a download's job, unless cancel() ended it."""
    with self._lock:
      if self._stops.pop(job_id, None) is not None:
        self._jobs.finish(job_id, status, error)


def _source(
    request: Request, path: str, plan: list[Any] | None
) -> dict[str, Any]:
  """Where a model came from: the commit is the snapshot it is in."""
  commit = next((info.commit_hash for info in plan or []), '')
  parts = pathlib.Path(path).parts
  if 'snapshots' in parts:
    index = len(parts) - 1 - parts[::-1].index('snapshots')
    if index + 1 < len(parts):
      commit = parts[index + 1]
  return {
      'repo': request.repo,
      'file': request.file,
      'revision': request.revision,
      'commit': commit,
  }


def _links(path: pathlib.Path) -> list[pathlib.Path]:
  """`path`, each link it leads through, and the file it ends at."""
  chain = [path]
  while chain[-1].is_symlink() and len(chain) < 8:
    target = pathlib.Path(os.readlink(chain[-1]))
    chain.append(target if target.is_absolute() else chain[-1].parent / target)
  return chain
