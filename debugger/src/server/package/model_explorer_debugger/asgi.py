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

"""Bounded asynchronous transport; all domain operations live in Application."""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import queue
import re
import sqlite3
import threading
import time
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from model_debugger_contracts.jobs import TERMINAL_STATUSES
from model_explorer_debugger import errors

from .analysis import AnalysisPool
from .api import Application
from .session_rules import UPLOAD_IDLE_TIMEOUT_SECONDS, UPLOAD_LIMIT_BYTES


class WorkspaceLease:
  """Exclusive coordinator ownership, taken before reading workspace state."""

  def __init__(self, root: str | Path) -> None:
    # fcntl is POSIX-only; importing it here keeps the module importable
    # where the lease is never taken.
    import fcntl  # pylint: disable=g-import-not-at-top

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    self.file = (root / '.server.lock').open('a+')
    try:
      fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
      self.file.close()
      raise RuntimeError(
          'Workspace already has an active server coordinator'
      ) from None

  def close(self) -> None:
    self.file.close()


class UploadStream:
  """A bounded bridge from ASGI chunks to the existing durable file writer."""

  def __init__(self) -> None:
    self.queue: queue.Queue[bytes | None] = queue.Queue(maxsize=4)
    self.aborted = threading.Event()
    self.finished = threading.Event()
    self.buffer = bytearray()
    self.eof = False

  def push(self, chunk: bytes | None) -> None:
    while not self.finished.is_set() and not self.aborted.is_set():
      try:
        self.queue.put(chunk, timeout=0.1)
        return
      except queue.Full:
        pass
    raise ValueError('Upload interrupted')

  def read(self, size: int) -> bytes:
    while len(self.buffer) < size and not self.eof:
      if self.aborted.is_set():
        raise ValueError('Upload interrupted')
      try:
        chunk = self.queue.get(timeout=0.1)
      except queue.Empty:
        continue
      if chunk is None:
        self.eof = True
      else:
        self.buffer.extend(chunk)
    result = bytes(self.buffer[:size])
    del self.buffer[:size]
    return result


async def upload(request: Request, metadata: Any) -> dict[str, Any]:
  """Streams an octet-stream request body into `metadata.upload`.

  Args:
    request: The Starlette request carrying the artifact bytes.
    metadata: The session metadata service that durably writes the file.

  Returns:
    The upload result reported by `metadata.upload`.

  Raises:
    ValueError: The request is not a bounded artifact upload, or the stream
      stalled, overran, or ended before Content-Length bytes arrived.
  """
  size = int(request.headers.get('content-length', '0'))
  if (
      request.headers.get('content-type', '').split(';')[0]
      != 'application/octet-stream'
      or not 0 < size <= UPLOAD_LIMIT_BYTES
  ):
    raise ValueError(
        'Expected an artifact file of at most'
        f' {UPLOAD_LIMIT_BYTES // 1024**3} GiB'
    )
  bridge = UploadStream()

  def consume() -> dict[str, Any]:
    try:
      return metadata.upload(
          bridge, size, unquote(request.headers.get('x-file-name', ''))
      )
    finally:
      bridge.finished.set()

  consumer = asyncio.create_task(asyncio.to_thread(consume))

  async def produce() -> None:
    received = 0
    tail = None
    # Large artifacts may legitimately take much longer than a minute; only a
    # stalled stream (no bytes for UPLOAD_IDLE_TIMEOUT_SECONDS) is interrupted.
    chunks = request.stream().__aiter__()
    while True:
      try:
        chunk = await asyncio.wait_for(
            chunks.__anext__(), timeout=UPLOAD_IDLE_TIMEOUT_SECONDS
        )
      except StopAsyncIteration:
        break
      except asyncio.TimeoutError:
        raise ValueError(
            'Upload stalled; no data received for'
            f' {UPLOAD_IDLE_TIMEOUT_SECONDS} seconds'
        )
      received += len(chunk)
      if received > size:
        raise ValueError('Upload exceeds Content-Length')
      for offset in range(0, len(chunk), 65536):
        if tail is not None:
          await asyncio.to_thread(bridge.push, tail)
        tail = chunk[offset : offset + 65536]
    if received != size:
      raise ValueError('Upload interrupted')
    if tail is not None:
      await asyncio.to_thread(bridge.push, tail)

  try:
    await produce()
    return await consumer
  finally:
    bridge.aborted.set()
    # Drain the writer so its partial-file cleanup completes before returning.
    # A writer failure that ends the upload was already raised by
    # `await consumer`; any other one is a consequence of the abort above.
    try:
      await asyncio.shield(consumer)
    except Exception as error:  # pylint: disable=broad-exception-caught
      logging.getLogger(__name__).debug('Upload writer stopped: %r', error)


def create_app(
    store: Any = None,
    registry: Any = None,
    ui: str | Path | None = None,
    analysis: AnalysisPool | Any | None = None,
    lease: WorkspaceLease | None = None,
) -> FastAPI:
  """Builds the loopback-only FastAPI app around one `Application`.

  Args:
    store: The selected capture store, if any.
    registry: The Session registry, if any.
    ui: Directory of static UI files to serve, or None for an API-only server.
    analysis: The analysis pool; a new `AnalysisPool` when None.
    lease: A `WorkspaceLease` released when the app shuts down.

  Returns:
    The FastAPI application.
  """
  pool = analysis if analysis is not None else AnalysisPool()
  try:
    application = Application(store, registry, pool)
  except BaseException:
    pool.close()
    raise

  @asynccontextmanager
  async def lifespan(app):
    try:
      yield
    finally:
      try:
        await asyncio.to_thread(application.close)
      finally:
        if lease:
          lease.close()

  app = FastAPI(
      lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
  )
  app.state.application = application
  logger = logging.getLogger('model_explorer_debugger.http')
  logger.setLevel(logging.INFO)
  if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
  logger.propagate = False

  @app.middleware('http')
  async def boundary(request: Request, call_next):
    request_id = uuid.uuid4().hex
    start = time.monotonic()
    code = 500
    try:
      host = request.headers.get('host', '')
      if urlparse('http://' + host).hostname not in (
          '127.0.0.1',
          'localhost',
          '::1',
      ):
        response = JSONResponse({'error': 'loopback_host_required'}, 403)
      elif (
          request.method != 'GET'
          and request.headers.get('origin')
          and urlparse(request.headers['origin']).netloc != host
      ):
        response = JSONResponse({'error': 'origin_mismatch'}, 403)
      else:
        response = await call_next(request)
      code = response.status_code
      response.headers['X-Request-ID'] = request_id
      response.headers['Cache-Control'] = 'no-store'
      return response
    finally:
      logger.info(
          json.dumps(
              dict(
                  request_id=request_id,
                  method=request.method,
                  path=request.url.path,
                  status=code,
                  elapsed_ms=round((time.monotonic() - start) * 1000, 2),
              )
          )
      )

  async def dispatch(request: Request):
    path = request.url.path
    query = errors.RequestFields(parse_qs(request.url.query))
    cancelled = threading.Event()
    watcher = None

    async def watch_disconnect():
      while not cancelled.is_set():
        if await request.is_disconnected():
          cancelled.set()
          return
        await asyncio.sleep(0.1)

    try:
      if request.method == 'GET':
        if path in ('/api/health', '/api/diagnostics'):
          return JSONResponse(
              application.diagnostics()
              if path.endswith('diagnostics')
              else {'status': 'ok'}
          )
        if application.jobs and (
            match := re.fullmatch(r'/api/jobs/([^/]+)/events', path)
        ):
          identity = match[1]
          after = max(
              0,
              int(
                  request.headers.get('last-event-id')
                  or query.get('after', ['0'])[0]
              ),
          )
          await asyncio.to_thread(application.jobs.get, identity)

          async def events():
            nonlocal after
            deadline = time.monotonic() + 25
            while (
                time.monotonic() < deadline
                and not await request.is_disconnected()
            ):
              batch = await asyncio.to_thread(
                  application.jobs.events, identity, after
              )
              for event in batch:
                after = event['sequence']
                yield f'id: {after}\ndata: {json.dumps(event)}\n\n'
              job = await asyncio.to_thread(application.jobs.get, identity)
              if (
                  job['status'] in TERMINAL_STATUSES
                  and after >= job['sequence']
              ):
                break
              if batch:
                continue
              yield ': heartbeat\n\n'
              await asyncio.sleep(0.2)

          return StreamingResponse(events(), media_type='text/event-stream')
        if ui and not path.startswith('/api/'):
          root = Path(ui).resolve()
          candidate = (
              root / ('index.html' if path == '/' else path.lstrip('/'))
          ).resolve()
          if candidate.is_relative_to(root) and candidate.is_file():
            return FileResponse(candidate)
          raise errors.NotFound('not_found')
        watcher = asyncio.create_task(watch_disconnect())
        result = await asyncio.to_thread(
            application.get, path, query, cancelled.is_set
        )
      else:
        if path == '/api/artifacts/upload':
          return JSONResponse(await upload(request, application.metadata))
        limit = (
            32 * 1024**2
            if re.fullmatch(r'/api/sessions/[^/]+/turns', path)
            else 65536
        )
        size = int(request.headers.get('content-length', '0'))
        if (
            not 0 < size <= limit
            or request.headers.get('content-type', '').split(';')[0]
            != 'application/json'
        ):
          raise ValueError('invalid_json_request')

        async def body():
          data = bytearray()
          async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > size:
              raise ValueError('invalid_json_request')
          if len(data) != size:
            raise ValueError('invalid_json_request')
          return data

        try:
          data = await asyncio.wait_for(body(), 60)
        except asyncio.TimeoutError:
          return JSONResponse({'error': 'request_body_timeout'}, 408)
        payload = json.loads(data, object_hook=errors.RequestFields)
        if not isinstance(payload, dict):
          raise ValueError('Expected a JSON object')
        watcher = asyncio.create_task(watch_disconnect())
        result = await asyncio.to_thread(
            application.post, path, query, payload, cancelled.is_set
        )
      return JSONResponse(result)
    except errors.NotFound as error:
      return JSONResponse({'error': str(error)}, 404)
    except (errors.AnalysisBusy, sqlite3.Error) as error:
      return JSONResponse(
          {'error': str(error)}, 503, headers={'Retry-After': '1'}
      )
    except ValueError as error:
      # The domain contract for a request the client must change.
      return JSONResponse({'error': str(error)}, 400)
    except Exception:  # pylint: disable=broad-exception-caught
      logger.exception('Unhandled API error')
      return JSONResponse({'error': 'internal_server_error'}, 500)
    finally:
      cancelled.set()
      if watcher:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)

  app.add_api_route('/{path:path}', dispatch, methods=['GET', 'POST'])
  return app
