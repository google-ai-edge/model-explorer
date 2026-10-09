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

"""Repeatable local capture publication; never sends an execution command."""

from copy import deepcopy
import shutil
from .capture_importer import publish_capture
from .store import SessionStore


def resolve_publication(
    registry, publication, import_result, publish=publish_capture
):
  record = deepcopy(publication['record'])
  identity = publication['job_id']
  directory = registry.root / 'sessions' / record['id'] / 'jobs' / identity
  destination = (
      registry.root / 'sessions' / record['id'] / 'captures' / identity
  )
  for path in (directory, destination):
    if not path.resolve().is_relative_to(registry.root):
      raise ValueError('Capture publication path escapes workspace')
  try:
    if destination.exists():
      store = SessionStore(destination)
      if store.session.get('capture_id') != identity or not any(
          turn['n'] == publication['turn'] for turn in store.session['turns']
      ):
        raise ValueError('Published capture identity does not match its job')
      for tensor in store.tensors:
        store.load(tensor)
      for resource in store.telemetry()['resources']:
        store.load_resource(resource['id'])
      capture = str(destination.relative_to(registry.root))
    else:
      if not directory.is_dir():
        raise ValueError('Received capture directory is unavailable')
      temporary = destination.with_name(identity + '.pending')
      if temporary.exists():
        shutil.rmtree(temporary)
      results = deepcopy(publication['results'])
      for run in record['runs']:
        role = run['id']
        result = results[role]
        if (
            result.get('debug_data', {}).get('status') == 'unavailable'
            or result.get('dump_complete') is False
        ):
          raise ValueError(
              result.get('debug_data', {}).get('error')
              or result.get('dump_error')
              or 'Runner dump is unavailable'
          )
        if import_result:
          imported = import_result(publication['requests'][role], result)
          if imported is not None:
            results[role] = result = imported
          if result.get('debug_data', {}).get('status') == 'unavailable':
            raise ValueError(
                result['debug_data'].get('error', 'Local capture import failed')
            )
      capture = publish(
          registry,
          {**record, 'publication_turn': publication['turn']},
          directory,
          results,
          publication['artifacts'],
      )
    return {'status': 'available', 'capture': capture}
  except Exception as error:
    return {'status': 'unavailable', 'error': str(error)}
