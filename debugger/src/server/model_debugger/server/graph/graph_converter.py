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

"""Model Explorer's converter, run in a child process of its own.

The converter is the prebuilt one of Model Explorer, the PyPI package
ai-edge-model-explorer-adapter (ADAPTER_VERSION): a C library under ctypes
that reads a TFLite model and writes Model Explorer's graph JSON of it
(ConvertFlatbufferDirectlyToJson). It reads the whole model into memory and
unpacks it, so the child does not give it the model: it gives it, one
subgraph at a time, the structure copy of each subgraph a signature runs
(tap/structure_copy), the model without its weights. Converting Gemma 4
E2B's signatures so takes about 4 s and 0.7 GB, E4B's 5 s and 0.9 GB (the
whole prefill and decode section took 10 s and 4.7 GB, and 22 s and
8.1 GB). It runs in a child process all the same, so that the
memory is the server's again once it ends and a crash of the library ends
only the child. The library has wheels for macOS on arm64 and Linux on
x86_64 and aarch64 only; elsewhere the child says the converter is not
available (UNAVAILABLE).

The child keeps only each signature's graph, slimmed
(model_graph.signature_graphs()), and writes it as JSON; the converter's
logging, a line for every composite operator, goes to a file beside it
(log_path()), not to the server's log; so do the structure copies while it
converts them. Everything else the server says of a graph is its own, in
Python, and is worked out as the graph is read (model_graph.annotate()).

The child never outlives its conversion: convert() kills it, and waits for
it, when it is stopped or takes longer than its timeout; and the child ends
itself once the server's process is gone, killed or not: its stdin is a
pipe only the server holds, which then reads as ended, and it checks that
its parent is still the process that started it.

The converter is to move into this repository's C++ package later; then
this module runs that one instead.

Usage of the child: python -m model_debugger.server.graph.graph_converter
  MODEL.litertlm SUBGRAPHS.json OUT.json
"""

from collections.abc import Callable, Iterable, Mapping
import json
import os
import pathlib
import platform
import signal
import stat
import subprocess
import sys
import threading
import time

from model_debugger.server.graph import model_graph
from model_debugger.server.tap import litertlm_file
from model_debugger.server.tap import structure_copy

ADAPTER = 'ai-edge-model-explorer-adapter'
ADAPTER_VERSION = '0.2.0'
UNAVAILABLE = 'The graph converter is not available on this platform.'
# How often a conversion looks whether it was stopped or took too long, and
# the child whether its parent is gone.
_POLL_SECONDS = 0.25
# The last bytes of the child's log a failure quotes.
_LOG_TAIL = 2000
# What the child exits with once its parent is gone.
_PARENT_GONE = 3
# A child killed by SIGKILL, not by convert(), that had used this share of
# the computer's memory was likely killed for want of it.
_MOST_OF_MEMORY = 0.5


class StoppedError(Exception):
  """The conversion was stopped before it ended."""


# The old name, still used by runner/ and session/.
Stopped = StoppedError


def identity() -> str:
  """What names the graphs this converter and its slimming write.

  The adapter's version, what it converts (structure_copy.identity()), and
  a digest of the code of the slimming the child does before a graph is
  kept (model_graph.slimming_digest()): a graph any of them would now
  write otherwise is not reused.

  Returns:
      The adapter, what it converts, and the slimming's digest, in words.
  """
  return (
      f'{ADAPTER} {ADAPTER_VERSION} of {structure_copy.identity()}, '
      f'slimmed {model_graph.slimming_digest()}'
  )


def log_path(out: pathlib.Path) -> pathlib.Path:
  """Where convert() writes the child's log, given its `out`."""
  return out.with_name(out.name + '.log')


Converter = Callable[
    [
        pathlib.Path,
        Mapping[str, int],
        pathlib.Path,
        Callable[[], bool],
        float,
    ],
    None,
]


def convert(
    model: pathlib.Path,
    subgraphs: Mapping[str, int],
    out: pathlib.Path,
    stopped: Callable[[], bool],
    timeout: float,
) -> None:
  """Converts a .litertlm model's text signatures to their graphs, in a child.

  Returns, or raises, only once the child has ended.

  Args:
      model: The .litertlm model; its prefill and decode section is converted.
      subgraphs: The subgraph each signature runs, by its key.
      out: Where the graphs go, as JSON: model_graph.signature_graphs()'s
        answer. The child's log goes beside it (log_path()).
      stopped: Whether to stop; the child is killed then.
      timeout: The seconds the child may take; it is killed then.

  Raises:
      StoppedError: `stopped` said so.
      RuntimeError: The converter failed, crashed or took too long; the
        text says how.
  """
  error = out.with_name(out.name + '.error')
  environment = dict(os.environ)
  # The child imports what this process imports, from where it does.
  environment['PYTHONPATH'] = os.pathsep.join(p for p in sys.path if p)
  deadline = time.monotonic() + timeout
  with open(log_path(out), 'wb') as written:
    child = subprocess.Popen(
        [
            sys.executable,
            '-m',
            __name__,
            str(model),
            json.dumps(dict(subgraphs)),
            str(out),
        ],
        # Never written to: it reads as ended once this process is gone.
        stdin=subprocess.PIPE,
        stdout=written,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    usage = None
    try:
      while usage is None:
        if stopped():
          raise StoppedError()
        if time.monotonic() > deadline:
          raise RuntimeError(
              f'The converter did not finish in {timeout:g} s, and was stopped.'
          )
        time.sleep(_POLL_SECONDS)
        usage = _reaped(child, os.WNOHANG)
    finally:
      if usage is None:
        child.kill()
        _reaped(child, 0)
      child.stdin.close()
  if child.returncode == 0 and out.is_file():
    return
  if error.is_file():
    raise RuntimeError(error.read_text(errors='replace').strip())
  tail = log_path(out).read_bytes()[-_LOG_TAIL:].decode(errors='replace')
  last = tail.strip().splitlines()[-1] if tail.strip() else 'no output'
  raise RuntimeError(f'The converter {_ended(child.returncode, usage)}: {last}')


def _reaped(
    child: subprocess.Popen[bytes], options: int
) -> os.struct_result | None:
  """Waits for the child, as os.wait4(); its resource usage once ended.

  Sets the child's returncode, as Popen's own wait does.

  Args:
      child: The child process.
      options: os.wait4()'s options, such as os.WNOHANG.

  Returns:
      The child's resource usage once it has ended; None if it has not.
  """
  pid, status, usage = os.wait4(child.pid, options)
  if pid == 0:
    return None
  child.returncode = os.waitstatus_to_exitcode(status)
  return usage


def _ended(returncode: int, usage: os.struct_result | None) -> str:
  """How a child that failed ended, in words.

  A signal is named; a SIGKILL convert() did not send is said to be for
  want of memory only when the child had used most of it.

  Args:
      returncode: The child's returncode; negative for the signal it got.
      usage: The child's resource usage (_reaped()).

  Returns:
      How it ended, in words.
  """
  if returncode >= 0:
    return f'ended with status {returncode}'
  number = -returncode
  try:
    name = signal.Signals(number).name
  except ValueError:
    name = 'unknown'
  text = f'was killed by signal {number} ({name})'
  if number != signal.SIGKILL:
    return text
  # ru_maxrss is in bytes on macOS, in KiB on Linux.
  peak = usage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
  try:
    memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
  except (OSError, ValueError):
    return text
  if peak < _MOST_OF_MEMORY * memory:
    return text
  return (
      f"{text} after using {peak / 1024**3:.1f} GB of this computer's "
      f'{memory / 1024**3:.1f} GB (out of memory?)'
  )


class _UnavailableError(Exception):
  """The converter cannot be imported on this platform."""


def _convert_here(
    model: pathlib.Path, subgraphs: Mapping[str, int], out: pathlib.Path
) -> None:
  """The child's work: converts, keeps the signatures' graphs, writes them.

  The structure copies are written first, beside `out`, and the model let
  go of; then each is converted, its graphs kept, and it is deleted.

  Args:
      model: The .litertlm model.
      subgraphs: The subgraph each signature runs, by its key.
      out: Where the graphs go, as JSON.

  Raises:
      _UnavailableError: The converter cannot be imported on this platform.
  """
  # Only the child loads the library.
  try:
    import ai_edge_model_explorer_adapter as adapter  # pylint: disable=g-import-not-at-top
  except ImportError as error:
    raise _UnavailableError(
        f'{UNAVAILABLE} ({ADAPTER} {ADAPTER_VERSION} on {sys.platform} '
        f'{platform.machine()}: {error})'
    ) from error

  copies = _write_copies(model, subgraphs.values(), out)
  try:
    graphs = {}
    for index, path in copies.items():
      text = adapter.ConvertFlatbufferDirectlyToJson(
          adapter.VisualizeConfig(), str(path)
      )
      path.unlink()
      graphs.update(
          model_graph.signature_graphs(
              json.loads(text),
              {
                  name: subgraph
                  for name, subgraph in subgraphs.items()
                  if subgraph == index
              },
          )
      )
      del text
  finally:
    for path in copies.values():
      path.unlink(missing_ok=True)
  pending = out.with_name(out.name + '.pending')
  with open(pending, 'w', encoding='utf-8') as file:
    json.dump(
        {name: graphs[name] for name in subgraphs},
        file,
        separators=(',', ':'),
    )
  os.replace(pending, out)


def _write_copies(
    model: pathlib.Path, subgraphs: Iterable[int], out: pathlib.Path
) -> dict[int, pathlib.Path]:
  """Writes the structure copy of each subgraph beside `out`.

  Args:
      model: The .litertlm model.
      subgraphs: The subgraphs' indices.
      out: The path the copies are written beside.

  Returns:
      Each subgraph's copy, by its index, in the order first named.
  """
  written = {}
  try:
    with litertlm_file.mapped(model) as container:
      # Not released: an error's traceback may hold views of it, and
      # the map closes once the last view goes.
      data = container.bytes_of(container.prefill_decode())
      for index, body in structure_copy.copies(data, subgraphs):
        path = out.with_name(f'{out.name}.subgraph-{index}')
        written[index] = path
        path.write_bytes(body)
      del data
  except BaseException:
    for path in written.values():
      path.unlink(missing_ok=True)
    raise
  return written


def _end_with_parent() -> None:
  """Ends the child once its parent is gone, watched on threads of its own.

  The library releases the interpreter while it converts (ctypes.CDLL), so
  the threads run meanwhile. Its stdin is read by its descriptor, not by
  sys.stdin, whose lock a thread blocked in it would hold as the
  interpreter exits.
  """
  parent = os.getppid()
  stdin = sys.stdin.fileno()

  def input_ended() -> None:
    while os.read(stdin, 65536):
      pass
    os._exit(_PARENT_GONE)

  def reparented() -> None:
    while os.getppid() == parent:
      time.sleep(_POLL_SECONDS)
    os._exit(_PARENT_GONE)

  if stat.S_ISFIFO(os.fstat(stdin).st_mode):
    threading.Thread(target=input_ended, daemon=True).start()
  threading.Thread(target=reparented, daemon=True).start()


def main(argv: list[str]) -> int:
  """The child: converts argv's model; a failure's text goes beside out."""
  if len(argv) != 4:
    print('\n'.join(__doc__.strip().splitlines()[-2:]), file=sys.stderr)
    return 2
  _end_with_parent()
  model, subgraphs, out = pathlib.Path(argv[1]), argv[2], argv[3]
  out = pathlib.Path(out)
  error = out.with_name(out.name + '.error')
  try:
    _convert_here(model, json.loads(subgraphs), out)
  except _UnavailableError as unavailable:
    error.write_text(str(unavailable))
    return 1
  # The child's top level: anything the converter raises is reported to the
  # parent as text, never as a traceback.
  except Exception as failure:  # pylint: disable=broad-exception-caught
    error.write_text(f'The converter failed: {failure}')
    return 1
  return 0


if __name__ == '__main__':
  sys.exit(main(sys.argv))
