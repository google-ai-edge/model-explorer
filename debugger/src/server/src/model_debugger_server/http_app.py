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

"""The HTTP API the UI calls, and the UI's own files.

Every route the UI can reach is here; any other /api path answers 404. The
server listens on loopback only and answers only requests that name a
loopback host, so a web page elsewhere cannot drive it.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
import functools
import json
import logging
import pathlib
import urllib.parse

from model_debugger_server import capture_view, errors
from model_debugger_server import execution as execution_lib
from model_debugger_server import jobs as jobs_lib
from model_debugger_server import models as models_lib
from model_debugger_server import sessions as sessions_lib
from model_debugger_server.analysis import kv, tokens
from starlette import (
    applications,
    datastructures,
    requests,
    responses,
    routing,
    types,
)

LOOPBACK_HOSTS = ('127.0.0.1', 'localhost', '::1')
KIB = 1024
MIB = 1024 * KIB
# JSON bodies: small, except a turn's prompt and the alignments the token
# and KV analyses carry.
BODY_LIMIT = 64 * KIB
TURN_BODY_LIMIT = 32 * MIB
ANALYSIS_BODY_LIMIT = 8 * MIB
# An event stream ends after this long; the browser reconnects by itself.
STREAM_SECONDS = 25
HEARTBEAT_SECONDS = 1
REQUEST_ID_LIMIT = 120

_logger = logging.getLogger(__name__)

Handler = Callable[[requests.Request], Awaitable[responses.Response]]


class _TooLarge(Exception):
  """The request body is larger than its route takes: 413."""


def _error(status: int, message: str) -> responses.JSONResponse:
  return responses.JSONResponse({'error': message}, status_code=status)


def _message(error: Exception) -> str:
  # A KeyError's str() is the key in quotes; its argument reads better.
  if isinstance(error, KeyError) and error.args:
    return f'Missing {error.args[0]}.'
  return str(error)


def _answers(handler: Handler) -> Handler:
  """Turns the errors a handler raises into the UI's {"error": ...} bodies."""

  @functools.wraps(handler)
  async def endpoint(request: requests.Request) -> responses.Response:
    try:
      return await handler(request)
    except errors.NotFound as error:
      return _error(404, str(error))
    except errors.Conflict as error:
      return _error(409, str(error))
    except _TooLarge as error:
      return _error(413, str(error))
    except execution_lib.Failed as error:
      return _error(502, str(error))
    except (ValueError, KeyError, TypeError) as error:
      return _error(400, _message(error))
    except Exception:
      _logger.exception('%s %s failed', request.method, request.url.path)
      return _error(500, 'internal_server_error')

  return endpoint


async def _json(request: requests.Request, limit: int = BODY_LIMIT) -> dict:
  """The request's JSON object body, read up to limit bytes."""
  content_type = request.headers.get('content-type', '')
  if content_type.split(';')[0].strip() != 'application/json':
    raise ValueError('Send a JSON object as application/json.')
  body = bytearray()
  async for chunk in request.stream():
    body += chunk
    if len(body) > limit:
      raise _TooLarge(f'The request body is larger than {limit} bytes.')
  payload = json.loads(body or b'null')
  if not isinstance(payload, dict):
    raise ValueError('Send a JSON object.')
  return payload


def _request_id(payload: dict) -> str:
  request_id = payload.get('request_id')
  if not isinstance(request_id, str) or not (
      0 < len(request_id) <= REQUEST_ID_LIMIT
  ):
    raise ValueError(
        f'request_id must be a string of 1 to {REQUEST_ID_LIMIT} characters.'
    )
  return request_id


def _id(payload: dict) -> str:
  record_id = payload.get('id')
  if not isinstance(record_id, str) or not record_id:
    raise ValueError('Name the Session by its id.')
  return record_id


def _hostname(host: str) -> str:
  """The name part of a Host header: 'a:1' -> 'a', '[::1]:1' -> '::1'."""
  if host.startswith('['):
    return host[1 : host.find(']')] if ']' in host else host
  return host.rsplit(':', 1)[0] if host.count(':') == 1 else host


class _Guard:
  """Answers only loopback hosts and same-origin writes; never cached."""

  def __init__(self, app: types.ASGIApp) -> None:
    self._app = app

  async def __call__(
      self, scope: types.Scope, receive: types.Receive, send: types.Send
  ) -> None:
    if scope['type'] != 'http':
      await self._app(scope, receive, send)
      return
    headers = datastructures.Headers(scope=scope)
    host = headers.get('host', '')
    refusal = None
    if _hostname(host) not in LOOPBACK_HOSTS:
      refusal = 'loopback_host_required'
    elif headers.get('sec-fetch-site') == 'cross-site':
      refusal = 'origin_mismatch'
    elif scope['method'] not in ('GET', 'HEAD'):
      origin = headers.get('origin')
      if origin is not None and urllib.parse.urlsplit(origin).netloc != host:
        refusal = 'origin_mismatch'
    if refusal is not None:
      await _error(403, refusal)(scope, receive, send)
      return

    async def send_uncached(message: types.Message) -> None:
      if message['type'] == 'http.response.start':
        datastructures.MutableHeaders(scope=message)['Cache-Control'] = (
            'no-store'
        )
      await send(message)

    await self._app(scope, receive, send_uncached)


def create_app(
    executions: execution_lib.Executions,
    sessions: sessions_lib.SessionStore,
    models: models_lib.ModelStore,
    jobs: jobs_lib.Jobs,
    ui: pathlib.Path | None,
) -> types.ASGIApp:
  """The server's ASGI application.

  Args:
    executions: every Session's Model Server.
    sessions: the Session and Chat records.
    models: the model files.
    jobs: the jobs and their events.
    ui: the built UI's directory, or None to serve the API alone.

  Returns:
    The application, guarded for loopback use.
  """
  run = asyncio.to_thread

  def listing() -> dict:
    capabilities = executions.capabilities()
    return {
        'sessions': [executions.summary(record) for record in sessions.list()],
        'configuration_template': {
            'id': '',
            'name': '',
            'model': '',
            'created_at': None,
            'status': 'draft',
            'has_capture': False,
            'notice': '',
            'tap_profile': '',
            'tap_points': {},
            'runs': [
                {
                    'id': side,
                    'runtime': 'LiteRT-LM',
                    'artifact': '',
                    'source': 'upload',
                    'device': 'local',
                    'backend': 'CPU',
                    'contextLength': '4096',
                }
                for side in execution_lib.SIDES
            ],
        },
        'capabilities': {
            'create': True,
            'rename': True,
            'duplicate': True,
            'delete': True,
        },
        'unavailable_reason': '',
        'generation_available': capabilities['available'],
    }

  async def health(request: requests.Request) -> responses.Response:
    return responses.JSONResponse({'status': 'ok'})

  async def get_sessions(request: requests.Request) -> responses.Response:
    return responses.JSONResponse(await run(listing))

  def summarized(change: Callable[[dict], dict]) -> Handler:
    async def handler(request: requests.Request) -> responses.Response:
      payload = await _json(request)
      record = await run(lambda: executions.summary(change(payload)))
      return responses.JSONResponse(record)

    return _answers(handler)

  def update(payload: dict) -> dict:
    executions.require_idle(_id(payload))
    return sessions.update(payload)

  def delete(payload: dict) -> dict:
    executions.require_idle(_id(payload))
    return sessions.delete(_id(payload))

  def configure_chat(payload: dict) -> dict:
    executions.require_idle(_id(payload))
    return sessions.set_generation(_id(payload), payload.get('generation'))

  @_answers
  async def new_chat(request: requests.Request) -> responses.Response:
    payload = await _json(request)
    record = await run(
        executions.new_chat,
        _id(payload),
        payload.get('name'),
        payload.get('generation'),
    )
    return responses.JSONResponse(record)

  @_answers
  async def initialize(request: requests.Request) -> responses.Response:
    payload = await _json(request)
    job = await run(
        executions.initialize,
        request.path_params['id'],
        _request_id(payload),
    )
    return responses.JSONResponse(job)

  @_answers
  async def turns(request: requests.Request) -> responses.Response:
    payload = await _json(request, TURN_BODY_LIMIT)
    job = await run(
        executions.generate,
        request.path_params['id'],
        _request_id(payload),
        payload.get('prompt'),
    )
    return responses.JSONResponse(job)

  @_answers
  async def close(request: requests.Request) -> responses.Response:
    await _json(request)
    record = await run(executions.close, request.path_params['id'])
    return responses.JSONResponse(record)

  @_answers
  async def get_job(request: requests.Request) -> responses.Response:
    return responses.JSONResponse(jobs.get(request.path_params['id']))

  @_answers
  async def cancel_job(request: requests.Request) -> responses.Response:
    await _json(request)
    job = await run(executions.cancel, request.path_params['id'])
    return responses.JSONResponse(job)

  @_answers
  async def job_events(request: requests.Request) -> responses.Response:
    job_id = request.path_params['id']
    jobs.get(job_id)
    cursor = request.headers.get('last-event-id') or request.query_params.get(
        'after', '0'
    )
    try:
      after = int(cursor)
    except ValueError:
      raise ValueError('after must be an event sequence number.') from None
    return responses.StreamingResponse(
        _event_stream(jobs, job_id, after),
        media_type='text/event-stream',
    )

  @_answers
  async def capabilities(request: requests.Request) -> responses.Response:
    return responses.JSONResponse(await run(executions.capabilities))

  @_answers
  async def devices(request: requests.Request) -> responses.Response:
    return responses.JSONResponse(await run(executions.devices))

  @_answers
  async def inspect(request: requests.Request) -> responses.Response:
    payload = await _json(request)
    return responses.JSONResponse(await run(executions.inspect, _id(payload)))

  @_answers
  async def upload(request: requests.Request) -> responses.Response:
    name = urllib.parse.unquote(request.headers.get('x-file-name', ''))
    if not name:
      raise ValueError('Name the file in X-File-Name.')
    length = request.headers.get('content-length', '')
    if not length.isdigit() or int(length) <= 0:
      raise ValueError('The upload needs its Content-Length.')
    size = int(length)
    if size > models_lib.UPLOAD_LIMIT_BYTES:
      raise _TooLarge(
          f'The file is larger than {models_lib.UPLOAD_LIMIT_BYTES} bytes.'
      )
    model = await models.upload(name, size, _idle_limited(request.stream()))
    return responses.JSONResponse(
        {'artifact': model.artifact, 'name': model.name, 'size': model.size}
    )

  def captured(
      analyze: Callable[..., dict], body_limit: int | None = None
  ) -> Handler:
    """A route over one record's capture, named by ?session_id=."""

    async def handler(request: requests.Request) -> responses.Response:
      query = dict(request.query_params)
      record_id = query.pop('session_id', '')
      if not record_id:
        raise ValueError('Name the capture by session_id.')
      payload = None
      if request.method == 'POST':
        payload = await _json(request, body_limit or BODY_LIMIT)

      def answer() -> dict:
        return analyze(executions.capture(record_id), record_id, query, payload)

      return responses.JSONResponse(await run(answer))

    return _answers(handler)

  def session(index, record_id, query, payload):
    return capture_view.session_view(index, sessions.get(record_id))

  def body_only(function):
    return lambda index, record_id, query, payload: function(index, payload)

  def query_only(function):
    return lambda index, record_id, query, payload: function(index, query)

  def query_and_body(function):
    return lambda index, record_id, query, payload: function(
        index, query, payload
    )

  api = [
      routing.Route('/api/health', health),
      routing.Route('/api/sessions', _answers(get_sessions)),
      routing.Route(
          '/api/sessions/create',
          summarized(sessions.create),
          methods=['POST'],
      ),
      routing.Route(
          '/api/sessions/update', summarized(update), methods=['POST']
      ),
      routing.Route(
          '/api/sessions/rename',
          summarized(lambda p: sessions.rename(_id(p), p.get('name'))),
          methods=['POST'],
      ),
      routing.Route(
          '/api/sessions/delete', summarized(delete), methods=['POST']
      ),
      routing.Route(
          '/api/sessions/restore',
          summarized(lambda p: sessions.restore(_id(p))),
          methods=['POST'],
      ),
      routing.Route('/api/sessions/chat', new_chat, methods=['POST']),
      routing.Route(
          '/api/sessions/chat-config',
          summarized(configure_chat),
          methods=['POST'],
      ),
      routing.Route(
          '/api/sessions/{id}/initialize', initialize, methods=['POST']
      ),
      routing.Route('/api/sessions/{id}/turns', turns, methods=['POST']),
      routing.Route('/api/sessions/{id}/close', close, methods=['POST']),
      routing.Route('/api/jobs/{id}', get_job),
      routing.Route('/api/jobs/{id}/events', job_events),
      routing.Route('/api/jobs/{id}/cancel', cancel_job, methods=['POST']),
      routing.Route('/api/runtime/capabilities', capabilities),
      routing.Route('/api/devices', devices),
      routing.Route('/api/devices/inspect', inspect, methods=['POST']),
      routing.Route('/api/artifacts/upload', upload, methods=['POST']),
      routing.Route('/api/session', captured(session)),
      routing.Route(
          '/api/token-analysis',
          captured(body_only(tokens.token_analysis), ANALYSIS_BODY_LIMIT),
          methods=['POST'],
      ),
      routing.Route(
          '/api/final-logits',
          captured(body_only(tokens.final_logits)),
          methods=['POST'],
      ),
      routing.Route(
          '/api/final-logits-vector',
          captured(body_only(tokens.final_logits_vector)),
          methods=['POST'],
      ),
      routing.Route('/api/kv-analysis', captured(query_only(kv.kv_analysis))),
  ]
  for path, analyze in (
      ('kv-range', kv.kv_range),
      ('kv-selection', kv.kv_selection),
      ('kv-head', kv.kv_head),
  ):
    api.append(
        routing.Route(
            f'/api/{path}',
            captured(query_and_body(analyze), ANALYSIS_BODY_LIMIT),
            methods=['GET', 'POST'],
        )
    )

  async def not_found(request: requests.Request) -> responses.Response:
    return _error(404, 'not_found')

  async def static(request: requests.Request) -> responses.Response:
    if ui is None:
      return _error(404, 'not_found')
    root = ui.resolve()
    path = (root / (request.path_params['path'] or 'index.html')).resolve()
    if not path.is_relative_to(root) or not path.is_file():
      return _error(404, 'not_found')
    return responses.FileResponse(path)

  api_methods = ['GET', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE']
  app = applications.Starlette(
      routes=[
          *api,
          routing.Route('/api/{path:path}', not_found, methods=api_methods),
          routing.Route('/{path:path}', static, methods=['GET', 'HEAD']),
      ]
  )
  return _Guard(app)


async def _event_stream(
    jobs: jobs_lib.Jobs, job_id: str, after: int
) -> AsyncIterator[str]:
  """A job's events as server-sent events, for at most STREAM_SECONDS.

  Frames are unnamed (the UI reads only onmessage) and carry the sequence as
  their id, so a reconnecting browser resumes with Last-Event-ID. While no
  event comes, a comment keeps the connection alive.
  """
  loop = asyncio.get_running_loop()
  deadline = loop.time() + STREAM_SECONDS
  events = jobs.events(job_id, after)
  pending = None
  try:
    while (remaining := deadline - loop.time()) > 0:
      if pending is None:
        pending = asyncio.ensure_future(anext(events))
      done, _ = await asyncio.wait(
          {pending}, timeout=min(HEARTBEAT_SECONDS, remaining)
      )
      if not done:
        yield ': heartbeat\n\n'
        continue
      try:
        event = pending.result()
      except StopAsyncIteration:
        return
      pending = None
      yield f'id: {event["sequence"]}\ndata: {json.dumps(event)}\n\n'
  finally:
    if pending is not None:
      pending.cancel()
      try:
        await pending
      except (asyncio.CancelledError, StopAsyncIteration):
        pass
    await events.aclose()


async def _idle_limited(
    chunks: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
  """The upload's bytes; stops when none arrive for the idle timeout."""
  iterator = aiter(chunks)
  while True:
    try:
      chunk = await asyncio.wait_for(
          anext(iterator), models_lib.UPLOAD_IDLE_TIMEOUT_SECONDS
      )
    except StopAsyncIteration:
      return
    except TimeoutError:
      raise ValueError(
          'No data arrived for '
          f'{models_lib.UPLOAD_IDLE_TIMEOUT_SECONDS} seconds; the upload '
          'stopped.'
      ) from None
    if chunk:
      yield chunk
