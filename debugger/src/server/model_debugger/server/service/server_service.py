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

"""The service the Server runs: the HTTP API the UI calls, and its files.

Every route the UI can reach is here; any other /api path answers 404. Any
other path is the UI's: one of its files, or else its index.html, since the UI
names its own pages by path (/sessions/<id>/debug/turns/2, say) and a page
opened or reloaded at such an address starts there. The server listens on
loopback only and answers only requests that name a loopback host, so a web
page elsewhere cannot drive it. The service owns the workspace and every
store in it, from create() until close().

The access log, at INFO, has a line for each request that changes something
and, at WARNING, for each that fails (logs_access()); the UI's polls and the
reads of a page as it is shown have none.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Collection
import concurrent.futures
import functools
import json
import pathlib
from typing import Any
import urllib.parse

from absl import logging
from model_debugger.server.analysis import tensor_analysis
from model_debugger.server.runner import launcher
from model_debugger.server.service import analysis_requests
from model_debugger.server.service import device_catalog
from model_debugger.server.service import graph_view
from model_debugger.server.service import metric_catalog
from model_debugger.server.service import pair_values
from model_debugger.server.service import session_listing
from model_debugger.server.service import session_view
from model_debugger.server.service import tap_view
from model_debugger.server.session import anchor_points
from model_debugger.server.session import capture_store
from model_debugger.server.session import hf_models
from model_debugger.server.session import jobs as jobs_lib
from model_debugger.server.session import mappings
from model_debugger.server.session import model_comparison
from model_debugger.server.session import model_graphs
from model_debugger.server.session import model_servers as model_servers_lib
from model_debugger.server.session import model_store
from model_debugger.server.session import op_menu
from model_debugger.server.session import prepared_models
from model_debugger.server.session import session_archive
from model_debugger.server.session import session_store
from model_debugger.server.session import tap_scans
from model_debugger.server.session import workspace
from model_debugger.server.util import errors
from starlette import applications
from starlette import datastructures
from starlette import requests
from starlette import responses
from starlette import routing
from starlette import types

# The most points GET /api/artifacts/{id}/counterparts pairs at once.
_MAX_COUNTERPARTS = 256

LOOPBACK_HOSTS = ('127.0.0.1', 'localhost', '::1')
KIB = 1024
MIB = 1024 * KIB
# JSON bodies: small, except a turn's prompt and the rows a match rate names.
BODY_LIMIT = 64 * KIB
TURN_BODY_LIMIT = 32 * MIB
ANALYSIS_BODY_LIMIT = 8 * MIB
# An event stream ends after this long; the browser reconnects by itself.
STREAM_SECONDS = 25
HEARTBEAT_SECONDS = 1

Handler = Callable[[requests.Request], Awaitable[responses.Response]]

# The analyses a page asks for by POST, with the body each takes; each only
# reads.
_POSTED_ANALYSES = (
    ('token-analysis', analysis_requests.token_analysis, None),
    ('token-range', analysis_requests.token_range, None),
    ('final-logits', analysis_requests.final_logits, None),
    ('final-logits-summary', analysis_requests.final_logits_summary, None),
    ('final-logits-bins', analysis_requests.final_logits_bins, None),
    ('final-logits-points', analysis_requests.final_logits_points, None),
    ('match-rate', analysis_requests.match_rate, ANALYSIS_BODY_LIMIT),
)
# The POSTs that only read, as reads by GET are: no access log line.
_READ_POSTS = frozenset(f'/api/{path}' for path, _, _ in _POSTED_ANALYSES)
# What a client that gave up on its request is answered.
_ABANDONED = 499
# What a request that failed unexpectedly is answered; its traceback is
# logged where it failed.
_INTERNAL_ERROR = 500


def logs_access(method: str, path: str, status: int) -> bool:
  """Whether a request has a line in the access log.

  The UI polls the listing, the devices and its jobs, and reads each
  analysis as a page shows it: a line each would bury what matters. So a
  read that succeeds (a GET, a HEAD, or a POST of _READ_POSTS) has none; a
  request that changes something has one, and so does any that failed,
  except one the client gave up on (499) and one whose traceback is logged
  already (500).

  Args:
    method: The request's method.
    path: Its path, without the query.
    status: The status it was answered with.

  Returns:
    True if the request should have a line in the access log, False otherwise.
  """
  if status in (_ABANDONED, _INTERNAL_ERROR):
    return False
  if status >= 400:
    return True
  return method not in ('GET', 'HEAD') and path not in _READ_POSTS


def _log_access(method: str, path: str, status: int) -> None:
  if not logs_access(method, path, status):
    return
  if status >= 400:
    logging.warning('%s %s %d', method, path, status)
  else:
    logging.info('%s %s %d', method, path, status)


class _TooLargeError(Exception):
  """The request body is larger than its route takes: 413."""


class _Ending(responses.StreamingResponse):
  """A streamed response that calls `end` once it is over, however.

  Sent whole, given up by the client, or failed: `end` is called as the
  response's call returns or raises.
  """

  def __init__(self, end: Callable[[], None], *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._end = end

  async def __call__(
      self, scope: types.Scope, receive: types.Receive, send: types.Send
  ) -> None:
    try:
      await super().__call__(scope, receive, send)
    finally:
      self._end()


def _error(status: int, message: str) -> responses.JSONResponse:
  return responses.JSONResponse({'error': message}, status_code=status)


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
    except _TooLargeError as error:
      return _error(413, str(error))
    except pair_values.AbandonedError:
      return _error(_ABANDONED, 'client_closed_request')
    except (ValueError, KeyError, TypeError) as error:
      return _error(400, errors.message(error))
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception('%s %s failed', request.method, request.url.path)
      return _error(_INTERNAL_ERROR, 'internal_server_error')

  return endpoint


async def _json(
    request: requests.Request, limit: int = BODY_LIMIT
) -> dict[str, Any]:
  """The request's JSON object body, read up to limit bytes."""
  content_type = request.headers.get('content-type', '')
  if content_type.split(';')[0].strip() != 'application/json':
    raise ValueError('Send a JSON object as application/json.')
  body = bytearray()
  async for chunk in request.stream():
    body += chunk
    if len(body) > limit:
      raise _TooLargeError(f'The request body is larger than {limit} bytes.')
  payload = json.loads(body or b'null')
  if not isinstance(payload, dict):
    raise ValueError('Send a JSON object.')
  return payload


def _id(payload: dict[str, Any]) -> str:
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
  """Answers only loopback hosts and same-origin writes; never cached.

  It writes the access log too (logs_access()).
  """

  def __init__(self, app: types.ASGIApp) -> None:
    self._app = app

  async def __call__(
      self, scope: types.Scope, receive: types.Receive, send: types.Send
  ) -> None:
    if scope['type'] != 'http':
      await self._app(scope, receive, send)
      return
    method, path = scope['method'], scope['path']

    async def send_logged(message: types.Message) -> None:
      if message['type'] == 'http.response.start':
        _log_access(method, path, message['status'])
      await send(message)

    headers = datastructures.Headers(scope=scope)
    host = headers.get('host', '')
    refusal = None
    if _hostname(host) not in LOOPBACK_HOSTS:
      refusal = 'loopback_host_required'
    elif scope['method'] not in ('GET', 'HEAD'):
      origin = headers.get('origin')
      if origin is not None and urllib.parse.urlsplit(origin).netloc != host:
        refusal = 'origin_mismatch'
    if refusal is not None:
      await _error(403, refusal)(scope, receive, send_logged)
      return

    async def send_uncached(message: types.Message) -> None:
      if message['type'] == 'http.response.start':
        datastructures.MutableHeaders(scope=message)['Cache-Control'] = (
            'no-store'
        )
      await send_logged(message)

    await self._app(scope, receive, send_uncached)


def _collect_models(
    models: model_store.ModelStore,
    sessions: session_store.SessionStore,
    graphs: model_graphs.ModelGraphs | None,
    pairs: mappings.MappingStore | None,
    among: Collection[str] | None = None,
) -> None:
  """Deletes the models no Session names, their graphs and tensor pairs.

  Args:
      models: The models.
      sessions: The Sessions, whose runs name the models kept.
      graphs: The models' graphs, if kept.
      pairs: The tensor pairs made by hand, if kept: those of a model deleted go
        with it.
      among: As model_store.ModelStore.collect(): the models of a Session just
        deleted, or None for every model, as the server starts.
  """
  removed = sessions.collect_models(among)
  if removed or among is None:
    kept = {model.sha256 for model in models.list()}
    if graphs is not None:
      graphs.collect(kept)
    if pairs is not None:
      pairs.collect(kept)


class ServerService:
  """The server's ASGI application, guarded for loopback use."""

  @classmethod
  def create(
      cls,
      workspace_root: pathlib.Path,
      launchers: launcher.Devices,
      ui: pathlib.Path | None,
      graph_timeout: float = model_graphs.DEFAULT_TIMEOUT_SECONDS,
  ) -> 'ServerService':
    """Opens the workspace and its stores; launches no runner.

    Lists the devices, and from then on every few seconds until close().
    The models no Session names are deleted first, with their files,
    graphs and tensor pairs (_collect_models()).

    Args:
        workspace_root: The workspace directory; created if missing.
        launchers: Where runners run.
        ui: The built UI's directory, or None to serve the API alone.
        graph_timeout: The seconds converting one model's graph may take
          (model_graphs).

    Returns:
        The service, holding the workspace until close().

    Raises:
        RuntimeError: Another server holds the workspace.
    """
    space = workspace.Workspace(workspace_root)
    model_servers = devices = scans = graphs = None
    try:
      # What a server killed left running, before anything launches.
      launchers.record_in(space.runners)
      models = model_store.ModelStore(space)
      scans = tap_scans.TapScans(models)
      sessions = session_store.SessionStore(space, models, scans)
      jobs = jobs_lib.Jobs()
      captures = capture_store.CaptureStore(space, sessions)
      exports = session_archive.Exports()
      graphs = model_graphs.ModelGraphs(
          space, models, scans, jobs, graph_timeout
      )
      pairs = mappings.MappingStore(space)
      _collect_models(models, sessions, graphs, pairs)
      prepared = prepared_models.PreparedModels(
          space,
          models,
          sessions,
          jobs,
          scans,
          graphs,
          pairs,
      )
      # Every reader of the devices shares one kept listing of them.
      devices = device_catalog.DeviceList(launchers)
      devices.start()
      model_servers = model_servers_lib.ModelServers(
          sessions, models, jobs, devices, captures, prepared, graphs
      )
      catalog = device_catalog.DeviceCatalog(
          devices, models, sessions, model_servers
      )
      listing = session_listing.SessionListing(
          sessions,
          model_servers,
          jobs,
          captures,
          prepared,
          exports,
      )
      return cls(
          space,
          devices,
          model_servers,
          catalog,
          listing,
          captures,
          exports,
          sessions,
          models,
          jobs,
          scans,
          prepared,
          ui,
          graphs,
      )
    except BaseException:
      if model_servers is not None:
        model_servers.shutdown()
      launchers.stop_all()
      if devices is not None:
        devices.stop()
      if scans is not None:
        scans.close()
      if graphs is not None:
        graphs.close()
      space.close()
      raise

  def __init__(
      self,
      workspace_: workspace.Workspace,
      devices: device_catalog.DeviceList,
      model_servers: model_servers_lib.ModelServers,
      catalog: device_catalog.DeviceCatalog,
      listing: session_listing.SessionListing,
      captures: capture_store.CaptureStore,
      exports: session_archive.Exports,
      sessions: session_store.SessionStore,
      models: model_store.ModelStore,
      jobs: jobs_lib.Jobs,
      scans: tap_scans.TapScans,
      prepared: prepared_models.PreparedModels,
      ui: pathlib.Path | None,
      graphs: model_graphs.ModelGraphs | None = None,
  ) -> None:
    """Serves the stores given; close() stops the runners and the workspace.

    Args:
        workspace_: The workspace the stores keep their files in.
        devices: The devices, listed until close().
        model_servers: Every Session's Model Server.
        catalog: The devices.
        listing: The records as the UI lists them.
        captures: Each chat's capture.
        exports: The Sessions being exported, which the listing shows.
        sessions: The Session records.
        models: The model files.
        jobs: The jobs and their events.
        scans: The scans of the models' tensors.
        prepared: The copies of models that observe tensors.
        ui: The built UI's directory, or None to serve the API alone.
        graphs: The models' graphs; None: their routes answer 404.
    """
    self._workspace = workspace_
    self._scans = scans
    self._graphs = graphs
    self._devices = devices
    self._model_servers = model_servers
    run = asyncio.to_thread
    # The analysis routes' own threads: other routes never wait behind.
    self._analysis = concurrent.futures.ThreadPoolExecutor(
        analysis_requests.THREADS, thread_name_prefix='analysis'
    )

    async def get_sessions(request: requests.Request) -> responses.Response:
      del request  # Unused.
      return responses.JSONResponse(await run(listing.listing))

    def summarized(
        change: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> Handler:
      async def handler(request: requests.Request) -> responses.Response:
        payload = await _json(request)
        record = await run(lambda: listing.summary(change(payload)))
        return responses.JSONResponse(record)

      return _answers(handler)

    # A delete is for good: the record first, then the capture, then the
    # copies and the models no Session uses any more. A Session being
    # exported waits.
    def delete(payload: dict[str, Any]) -> dict[str, Any]:
      exports.check_not_exporting(_id(payload))
      record = model_servers.delete(_id(payload))
      captures.forget(record['id'])
      workspace_.remove_session_files(record['id'])
      prepared.collect()
      _collect_models(
          models,
          sessions,
          graphs,
          prepared.mappings,
          {run['artifact'] for run in record['runs']},
      )
      return record

    @_answers
    async def tap_scan(request: requests.Request) -> responses.Response:
      # POST starts the scan of a model unless it has one; GET says
      # where it is. Both answer the UI's TapScan.
      artifact = request.path_params['id']
      if request.method == 'POST':
        await _json(request)
        state = await run(scans.start, artifact)
      else:
        state = await run(scans.state, artifact)
      note = await run(prepared.disk_note, models.get(artifact))
      return responses.JSONResponse(tap_view.scan(state, note))

    @_answers
    async def tap_anchors(request: requests.Request) -> responses.Response:
      # The anchors a Session of a LiteRT-LM model captures by default,
      # found in its graph; they are not known until it is converted,
      # which this never does.
      model = await run(models.get, request.path_params['id'])
      if model.runtime != model_store.LITERT_LM:
        raise errors.NotFound(
            f'{model.name} is a {model.runtime} model: its runner '
            'captures its anchors by its own option.'
        )
      found = await run(anchor_points.find, graphs, scans, model)
      note = await run(prepared.disk_note, model)
      return responses.JSONResponse(tap_view.anchors(found, note))

    comparison = model_comparison.ModelComparison(models, graphs)
    menu = op_menu.OpMenu(models, graphs, scans)

    @_answers
    async def compare_models(
        request: requests.Request,
    ) -> responses.Response:
      # The Session editor's hint of its two models, ?ref= and
      # ?target=, by artifact: what of them compares, by the analyses'
      # own rules (model_comparison); it converts and runs nothing.
      found = await run(
          comparison.compare,
          request.query_params.get('ref', ''),
          request.query_params.get('target', ''),
      )
      return responses.JSONResponse(found)

    @_answers
    async def layer_ops(request: requests.Request) -> responses.Response:
      # The editor's list of the operators inside the layers of its two
      # models, ?ref= and ?target= by artifact, with those ticked,
      # ?ops=<id>,<id> (op_menu); it converts and runs nothing.
      ticked = [
          op for op in request.query_params.get('ops', '').split(',') if op
      ]
      found = await run(
          menu.menu,
          request.query_params.get('ref', ''),
          request.query_params.get('target', ''),
          ticked,
      )
      return responses.JSONResponse(found)

    @_answers
    async def tap_counterparts(
        request: requests.Request,
    ) -> responses.Response:
      # Which PyTorch module each point chosen of a LiteRT-LM model is
      # paired with in a Session beside ?pytorch=<its model>, for the
      # editor: ?points=<id>,<id> (prepared_models.counterparts()).
      model = await run(models.get, request.path_params['id'])
      pytorch = await run(models.get, request.query_params.get('pytorch', ''))
      if (
          model.runtime != model_store.LITERT_LM
          or pytorch.runtime != model_store.PYTORCH
      ):
        raise errors.NotFound(
            'Points of a LiteRT-LM model are paired with a PyTorch '
            "model's modules."
        )
      ids = [
          point_id
          for point_id in request.query_params.get('points', '').split(',')
          if point_id
      ]
      if len(ids) > _MAX_COUNTERPARTS:
        raise ValueError(
            f'At most {_MAX_COUNTERPARTS} points are paired at once.'
        )
      found = await run(prepared.counterparts, model, pytorch, ids)
      return responses.JSONResponse(found)

    def kept_graphs() -> model_graphs.ModelGraphs:
      if graphs is None:
        raise errors.NotFound('This server keeps no graphs.')
      return graphs

    @_answers
    async def graph_status(
        request: requests.Request,
    ) -> responses.Response:
      # Where the model's graph is; it is converted only as a Session
      # of the model is initialized, never here.
      store = kept_graphs()
      artifact = request.path_params['id']
      state = await run(store.state, artifact)
      facts = architecture = None
      if state.status == model_graphs.COMPLETED:
        model = await run(models.get, artifact)
        facts = await run(store.facts, model)
        architecture = await run(store.architecture, model)
      return responses.JSONResponse(
          graph_view.status(state, facts, architecture)
      )

    @_answers
    async def graph_signature(
        request: requests.Request,
    ) -> responses.Response:
      store = kept_graphs()
      model = await run(models.get, request.path_params['id'])
      return responses.JSONResponse(
          await run(
              graph_view.graph,
              store,
              model,
              request.path_params['signature'],
          )
      )

    @_answers
    async def architecture(
        request: requests.Request,
    ) -> responses.Response:
      store = kept_graphs()
      model = await run(models.get, request.path_params['id'])
      return responses.JSONResponse({
          'status': model_graphs.COMPLETED,
          **(await run(store.architecture, model)),
      })

    @_answers
    async def chats(request: requests.Request) -> responses.Response:
      # Starts a chat of the Session, {generation: {ref, target}}: its
      # job, as a turn's (model_servers.ModelServers.start_chat()).
      payload = await _json(request)
      job = await run(
          model_servers.start_chat,
          request.path_params['id'],
          payload.get('generation'),
      )
      return responses.JSONResponse(job)

    @_answers
    async def turns(request: requests.Request) -> responses.Response:
      # A turn of the live chat, which `chat` names: refused if
      # another started meanwhile. A reply's limit is the chat's, fixed
      # when it is created (generation.maxOutputTokens): a turn sends
      # only its prompt, {prompt, chat?} (model_server.check_prompt).
      payload = await _json(request, TURN_BODY_LIMIT)
      job = await run(
          model_servers.generate,
          request.path_params['id'],
          payload.get('prompt'),
          payload.get('chat'),
      )
      return responses.JSONResponse(job)

    @_answers
    async def close(request: requests.Request) -> responses.Response:
      await _json(request)
      session_id = request.path_params['id']

      def turn_off() -> dict[str, Any]:
        model_servers.close(session_id)
        return listing.summary(sessions.get(session_id))

      return responses.JSONResponse(await run(turn_off))

    @_answers
    async def get_job(request: requests.Request) -> responses.Response:
      return responses.JSONResponse(jobs.get(request.path_params['id']))

    @_answers
    async def cancel_job(request: requests.Request) -> responses.Response:
      await _json(request)
      job_id = request.path_params['id']
      # A download is cancelled by its own; a turn by its Model Server.
      if jobs.get(job_id)['operation'] == 'download':
        job = await run(huggingface.cancel, job_id)
      else:
        job = await run(model_servers.cancel, job_id)
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
      del request  # Unused.
      return responses.JSONResponse(await run(catalog.capabilities))

    @_answers
    async def metrics(request: requests.Request) -> responses.Response:
      del request  # Unused.
      return responses.JSONResponse(metric_catalog.catalog())

    @_answers
    async def list_devices(request: requests.Request) -> responses.Response:
      del request  # Unused.
      return responses.JSONResponse(await run(catalog.devices))

    @_answers
    async def builds(request: requests.Request) -> responses.Response:
      device_id = request.query_params.get('id', '')
      if not device_id:
        raise ValueError('Name the device in ?id=.')
      return responses.JSONResponse(await run(catalog.builds, device_id))

    @_answers
    async def chat_options(
        request: requests.Request,
    ) -> responses.Response:
      record_id = request.query_params.get('id', '')
      if not record_id:
        raise ValueError('Name the Session in ?id=.')
      return responses.JSONResponse(
          await run(catalog.chat_capabilities, record_id)
      )

    @_answers
    async def inspect(request: requests.Request) -> responses.Response:
      payload = await _json(request)
      return responses.JSONResponse(await run(catalog.inspect, _id(payload)))

    huggingface = hf_models.HfModels(models, jobs, workspace_.hf)

    @_answers
    async def fetch_huggingface(
        request: requests.Request,
    ) -> responses.Response:
      # A model named on Hugging Face: a download job, followed as any
      # job is (GET /api/jobs/{id}); its `model` once it completed.
      payload = await _json(request)
      return responses.JSONResponse(await run(huggingface.start, payload))

    @_answers
    async def check_huggingface(
        request: requests.Request,
    ) -> responses.Response:
      # Whether the server has a model named on Hugging Face, and its
      # size: {on_server, size, text} (hf_models.HfModels.check()).
      payload = await _json(request)
      return responses.JSONResponse(await run(huggingface.check, payload))

    @_answers
    async def upload(request: requests.Request) -> responses.Response:
      name = urllib.parse.unquote(request.headers.get('x-file-name', ''))
      if not name:
        raise ValueError('Name the file in X-File-Name.')
      length = request.headers.get('content-length', '')
      if not length.isdigit() or int(length) <= 0:
        raise ValueError('The upload needs its Content-Length.')
      size = int(length)
      if size > model_store.UPLOAD_LIMIT_BYTES:
        raise _TooLargeError(
            f'The file is larger than {model_store.UPLOAD_LIMIT_BYTES} bytes.'
        )
      model = await models.upload(name, size, _idle_limited(request.stream()))
      return responses.JSONResponse({
          'artifact': model.artifact,
          'name': model.name,
          'size': model.size,
      })

    archives = session_archive.SessionArchives(
        workspace_,
        sessions,
        models,
        scans,
        graphs,
        captures,
        model_servers,
        exports,
    )

    @_answers
    async def export(request: requests.Request) -> responses.Response:
      # A Session that has ended, with all its chats, as one file the
      # browser saves (session_archive); its files are read as it is
      # sent, and it is exporting until the stream ends, sent whole or
      # given up.
      exported = await run(archives.export, request.path_params['id'])
      return _Ending(
          exported.end,
          exported.chunks,
          media_type='application/x-tar',
          headers={
              'Content-Disposition': session_archive.content_disposition(
                  exported.name
              ),
              'Content-Length': str(exported.size),
          },
      )

    @_answers
    async def import_session(
        request: requests.Request,
    ) -> responses.Response:
      # The bytes of an exported Session, as the body; answers its
      # record as the listing shows it.
      length = request.headers.get('content-length', '')
      if not length.isdigit() or int(length) <= 0:
        raise ValueError('The upload needs its Content-Length.')
      size = int(length)
      if size > session_archive.IMPORT_LIMIT_BYTES:
        raise _TooLargeError(
            'The file is larger than '
            f'{session_archive.IMPORT_LIMIT_BYTES} bytes.'
        )
      try:
        record = await archives.import_file(
            size, _idle_limited(request.stream())
        )
      except requests.ClientDisconnect:
        # The browser cancelled the upload; what arrived is removed.
        logging.info('Import cancelled by the client')
        return _error(_ABANDONED, 'client_closed_request')
      return responses.JSONResponse(await run(listing.summary, record))

    def captured(
        analyze: Callable[..., dict[str, Any]],
        body_limit: int | None = None,
        threads: concurrent.futures.Executor | None = None,
    ) -> Handler:
      """A route over one chat's capture: ?session_id= and ?chat=.

      analyze is given the capture, the chat (`(record id, number)`),
      the rest of the query and the body; on `threads`, it is also told
      whether the client is gone.

      Args:
        analyze: Callable analyzing the chat capture.
        body_limit: Maximum request body size.
        threads: Thread executor to run analyze on.

      Returns:
        The request handler endpoint.
      """

      async def handler(request: requests.Request) -> responses.Response:
        query = dict(request.query_params)
        record_id = query.pop('session_id', '')
        if not record_id:
          raise ValueError('Name the capture by session_id.')
        chat = query.pop('chat', '')
        if not (chat.isascii() and chat.isdigit()):
          raise ValueError('Name the chat by its number, chat.')
        key = (record_id, int(chat))
        payload = None
        if request.method == 'POST':
          payload = await _json(request, body_limit or BODY_LIMIT)

        loop = asyncio.get_running_loop()

        def gone() -> bool:
          return asyncio.run_coroutine_threadsafe(
              request.is_disconnected(), loop
          ).result()

        def answer() -> dict[str, Any]:
          told = () if threads is None else (gone,)
          return analyze(
              captures.get(*key),
              key,
              query,
              payload,
              *told,
          )

        if threads is None:
          return responses.JSONResponse(await run(answer))
        return responses.JSONResponse(
            await loop.run_in_executor(threads, answer)
        )

      return _answers(handler)

    # A view keeps what its Session observes, from its models' scans.
    views = session_view.SessionViews(
        observed=lambda record_id: prepared.observed(sessions.get(record_id)),
        pieces=models.pieces,
    )

    def asked(record_id: str) -> dict[str, set[str] | None] | None:
      """The modules each side's runner was asked for as it started.

      As the last chat's capture says, every chat's runners being the
      ones the Session started; None before its first chat.

      Args:
        record_id: The Session identifier.

      Returns:
        Mapping of side name to set of asked modules, or None.
      """
      for chat in reversed(sessions.get(record_id)['chats']):
        try:
          index = captures.get(record_id, chat['n'])
        except (errors.NotFound, ValueError):
          continue
        return {
            side: tensor_analysis.asked_modules(entry.get('load') or {})
            for side, entry in index.sides.items()
        }
      return None

    @_answers
    async def manual_pairs(request: requests.Request) -> responses.Response:
      # The pairs made by hand of the two models of the Session
      # ?session_id= names, as it has them (prepared_models
      # .manual_pairs()): GET lists them; POST {ref, target, label}
      # saves one, each tensor {signature, node, output} of its side's
      # graph; DELETE /api/mappings/{pair} deletes one. Each answers
      # the list as it is then; what the Session's analyses observe
      # is worked out again.
      record_id = request.query_params.get('session_id', '')
      if not record_id:
        raise ValueError('Name the Session by session_id.')
      if request.method == 'POST':
        payload = await _json(request)

        def save() -> dict[str, Any]:
          return prepared.save_pair(
              sessions.get(record_id),
              payload.get('ref'),
              payload.get('target'),
              payload.get('label'),
              asked(record_id),
          )

        answer = await run(save)
        views.forget()
      elif request.method == 'DELETE':
        pair_id = request.path_params.get('pair', '')

        def delete_pair() -> dict[str, Any]:
          return prepared.delete_pair(
              sessions.get(record_id), pair_id, asked(record_id)
          )

        answer = await run(delete_pair)
        views.forget()
      else:
        answer = await run(
            lambda: prepared.manual_pairs(
                sessions.get(record_id), asked(record_id)
            )
        )
      return responses.JSONResponse(answer)

    def session(index, key, query, payload):
      del query, payload  # Unused.
      record_id, n = key
      return views.session(record_id, n, index, sessions.get(record_id))

    def compared(analyze):
      """An analysis of rows of the chat's comparison, kept in views."""
      return lambda index, key, query, payload, gone: analyze(
          index,
          analysis_requests.Comparison(
              lambda: views.compared(*key, index), gone
          ),
          query if payload is None else payload,
      )

    api = [
        routing.Route('/api/sessions', _answers(get_sessions)),
        # A Session starts as it is created, or is not created: no
        # record is written of one that cannot start now.
        routing.Route(
            '/api/sessions/create',
            summarized(model_servers.create),
            methods=['POST'],
        ),
        routing.Route(
            '/api/sessions/rename',
            summarized(lambda p: sessions.rename(_id(p), p.get('name'))),
            methods=['POST'],
        ),
        routing.Route(
            '/api/sessions/delete', summarized(delete), methods=['POST']
        ),
        routing.Route('/api/sessions/chat-options', chat_options),
        routing.Route('/api/sessions/import', import_session, methods=['POST']),
        routing.Route('/api/sessions/{id}/export', export),
        routing.Route('/api/sessions/{id}/chats', chats, methods=['POST']),
        routing.Route('/api/sessions/{id}/turns', turns, methods=['POST']),
        routing.Route('/api/sessions/{id}/close', close, methods=['POST']),
        routing.Route('/api/jobs/{id}', get_job),
        routing.Route('/api/jobs/{id}/events', job_events),
        routing.Route('/api/jobs/{id}/cancel', cancel_job, methods=['POST']),
        routing.Route('/api/runtime/capabilities', capabilities),
        routing.Route('/api/metrics', metrics),
        routing.Route('/api/devices', list_devices),
        routing.Route('/api/devices/builds', builds),
        routing.Route('/api/devices/inspect', inspect, methods=['POST']),
        routing.Route('/api/artifacts/upload', upload, methods=['POST']),
        routing.Route(
            '/api/models/huggingface', fetch_huggingface, methods=['POST']
        ),
        routing.Route(
            '/api/models/huggingface/check',
            check_huggingface,
            methods=['POST'],
        ),
        routing.Route('/api/models/compare', compare_models),
        routing.Route('/api/models/ops', layer_ops),
        routing.Route(
            '/api/artifacts/{id}/scan', tap_scan, methods=['GET', 'POST']
        ),
        routing.Route('/api/artifacts/{id}/anchors', tap_anchors),
        routing.Route('/api/artifacts/{id}/counterparts', tap_counterparts),
        routing.Route('/api/artifacts/{id}/graph', graph_status),
        routing.Route('/api/artifacts/{id}/graph/{signature}', graph_signature),
        routing.Route('/api/artifacts/{id}/architecture', architecture),
        routing.Route('/api/mappings', manual_pairs, methods=['GET', 'POST']),
        routing.Route('/api/mappings/{pair}', manual_pairs, methods=['DELETE']),
        routing.Route('/api/session', captured(session)),
        routing.Route(
            '/api/kv-analysis',
            captured(
                lambda index, key, query, payload: (
                    analysis_requests.kv_metadata(index, query)
                )
            ),
        ),
    ]
    for path, analyze, body_limit in _POSTED_ANALYSES:
      api.append(
          routing.Route(
              f'/api/{path}',
              captured(compared(analyze), body_limit, self._analysis),
              methods=['POST'],
          )
      )
    for path, analyze in (
        ('kv-range', analysis_requests.kv_range),
        ('kv-head', analysis_requests.kv_head),
        ('position-analysis', analysis_requests.position_analysis),
        ('position-range', analysis_requests.position_range),
        ('position-tensor', analysis_requests.position_tensor),
    ):
      api.append(
          routing.Route(
              f'/api/{path}',
              captured(compared(analyze), threads=self._analysis),
          )
      )

    async def not_found(request: requests.Request) -> responses.Response:
      # A route of the API asked with another method: 405, saying which.
      for route in api:
        match, _ = route.matches(request.scope)
        if match is routing.Match.PARTIAL:
          return responses.JSONResponse(
              {'error': 'method_not_allowed'},
              status_code=405,
              headers={'Allow': ', '.join(sorted(route.methods))},
          )
      return _error(404, 'not_found')

    async def static(request: requests.Request) -> responses.Response:
      if ui is None:
        return _error(404, 'not_found')
      root = ui.resolve()
      path = (root / (request.path_params['path'] or 'index.html')).resolve()
      if not path.is_relative_to(root) or not path.is_file():
        # A page of the UI, which reads its address itself; a name
        # with a '.' is of a file the UI lacks, never a page.
        name = request.path_params['path'].rsplit('/', 1)[-1]
        path = root / 'index.html'
        if '.' in name or not path.is_file():
          return _error(404, 'not_found')
      return responses.FileResponse(path)

    api_methods = ['GET', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE']
    app = applications.Starlette(
        routes=[
            *api,
            routing.Route('/api', not_found, methods=api_methods),
            routing.Route('/api/{path:path}', not_found, methods=api_methods),
            routing.Route('/{path:path}', static, methods=['GET', 'HEAD']),
        ]
    )
    self._app = _Guard(app)

  @property
  def workspace_root(self) -> pathlib.Path:
    """The workspace directory, resolved."""
    return self._workspace.root

  async def __call__(
      self, scope: types.Scope, receive: types.Receive, send: types.Send
  ) -> None:
    """Answers one ASGI connection."""
    await self._app(scope, receive, send)

  def close(self) -> None:
    """Stops every runner and the device listing; frees the workspace.

    Every runner launched, also one still starting, whatever Session it
    is of; what does not stop is killed (launcher.Devices.stop_all()).
    """
    self._model_servers.shutdown()
    self._devices.stop_all()
    self._devices.stop()
    self._scans.close()
    if self._graphs is not None:
      self._graphs.close()
    self._analysis.shutdown(wait=False, cancel_futures=True)
    self._workspace.close()


async def _event_stream(
    jobs: jobs_lib.Jobs, job_id: str, after: int
) -> AsyncIterator[str]:
  """A job's events as server-sent events, for at most STREAM_SECONDS.

  Frames are unnamed (the UI reads only onmessage) and carry the sequence as
  their id, so a reconnecting browser resumes with Last-Event-ID. While no
  event comes, a comment keeps the connection alive.

  Args:
    jobs: Jobs store.
    job_id: The job identifier.
    after: Sequence number to resume after.

  Yields:
    Server-sent event chunks.
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
          anext(iterator), model_store.UPLOAD_IDLE_TIMEOUT_SECONDS
      )
    except StopAsyncIteration:
      return
    except TimeoutError:
      raise ValueError(
          'No data arrived for '
          f'{model_store.UPLOAD_IDLE_TIMEOUT_SECONDS} seconds; the '
          'upload stopped.'
      ) from None
    if chunk:
      yield chunk
