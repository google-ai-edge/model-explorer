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

"""UI metadata for the configured capture; never rewrites captured evidence."""

from datetime import datetime, timezone
import hashlib
import json
from threading import RLock
from uuid import NAMESPACE_URL, uuid4, uuid5

from model_explorer_debugger import fsutil
from model_explorer_debugger import generation_config
from model_explorer_debugger import session_rules

__all__ = ['SessionMetadata']


class SessionMetadata:

  def __init__(self, store):
    self.store = store
    self.path = store.root / '.debugger-session.json'
    self.workspace_path = store.root / '.debugger-workspace.json'
    self.lock = RLock()
    fingerprint = (
        getattr(store, 'source_session_fingerprint', None)
        or hashlib.sha256(
            json.dumps(store.session, sort_keys=True).encode()
        ).hexdigest()
    )
    self.identity = str(
        uuid5(NAMESPACE_URL, 'model-debugger:saved-capture:' + fingerprint)
    )

  def _name(self):
    if self.path.exists():
      record = json.loads(self.path.read_text())
      if record.get('id') == self.identity:
        return self.validate_name(record['name'])
    session = self.store.session
    if isinstance(session.get('name'), str) and session['name'].strip():
      return session['name'].strip()
    precisions = ' vs '.join(
        str(run.get('precision', 'Unknown')) for run in session.get('runs', [])
    )
    return str(session.get('model', 'Saved session')) + (
        ' · ' + precisions if precisions else ''
    )

  @staticmethod
  def validate_name(value):
    return session_rules.validate_name(value)

  def summary(self):
    with self.lock:
      session = self.store.session
      created = session.get('created_at')
      try:
        if not isinstance(created, str):
          raise ValueError()
        datetime.fromisoformat(created.replace('Z', '+00:00'))
      except ValueError:
        created = None
      return {
          'id': self.identity,
          'name': self._name(),
          'created_at': created,
          'model': session.get('model', 'Unknown'),
          'status': 'saved',
          'has_capture': True,
          'runs': session.get('runs', []),
          'notice': session.get('notice', ''),
      }

  def upload(self, stream, size, name):
    return session_rules.upload_artifact(self.store.root, stream, size, name)

  def _workspace(self):
    if self.workspace_path.exists():
      result = json.loads(self.workspace_path.read_text())
      if result.get('capture_id') == self.identity:
        return result
    return {'capture_id': self.identity, 'drafts': [], 'deleted': []}

  def _write_workspace(self, workspace):
    fsutil.atomic_json(self.workspace_path, workspace)

  def listing(self):
    with self.lock:
      workspace = self._workspace()
      records = [self.summary(), *workspace['drafts']]
      return {
          'sessions': [
              r for r in records if r['id'] not in workspace['deleted']
          ],
          'configuration_template': self.summary(),
          'capabilities': {
              'create': True,
              'duplicate': True,
              'delete': True,
              'rename': True,
          },
          'unavailable_reason': '',
          'generation_available': False,
      }

  def _find(self, workspace, identity):
    if identity in workspace['deleted']:
      raise ValueError('Session was deleted. Restore it before editing.')
    if identity == self.identity:
      return self.summary()
    return next((r for r in workspace['drafts'] if r['id'] == identity), None)

  @staticmethod
  def _config(payload):
    return session_rules.validate_config(payload)

  def manage(self, operation, payload):
    with self.lock:
      if not isinstance(payload, dict):
        raise ValueError('Invalid session request.')
      workspace = self._workspace()
      if operation == 'create':
        record = {
            **self._config(payload),
            'id': str(uuid4()),
            'created_at': datetime.now(timezone.utc).isoformat(),
            'status': 'draft',
            'has_capture': False,
            'notice': (
                'Configuration saved. Runtime initialization is not connected;'
                ' no execution has been captured.'
            ),
        }
        workspace['drafts'].append(record)
      elif operation == 'restore':
        identity = payload.get('id')
        record = (
            self.summary()
            if identity == self.identity
            else next(
                (r for r in workspace['drafts'] if r['id'] == identity), None
            )
        )
        if not record or identity not in workspace['deleted']:
          raise ValueError('Deleted session not found.')
        workspace['deleted'].remove(identity)
      else:
        record = self._find(workspace, payload.get('id'))
        if not record:
          raise ValueError('Session not found.')
        if operation == 'delete':
          workspace['deleted'].append(record['id'])
        elif operation in ('duplicate', 'chat'):
          parent_id = record.get('parent_session_id', record['id'])
          record = {
              **record,
              'id': str(uuid4()),
              'name': self.validate_name(
                  payload.get('name', session_rules.copy_name(record['name']))
              ),
              'created_at': datetime.now(timezone.utc).isoformat(),
              'status': 'draft',
              'has_capture': False,
              'notice': (
                  'Configuration copied. Execution data is not duplicated.'
              ),
          }
          if operation == 'chat':
            record['parent_session_id'] = parent_id
            if 'generation' in payload:
              record['generation'] = generation_config.generation_config(
                  payload['generation']
              )
          else:
            record.pop('parent_session_id', None)
          workspace['drafts'].append(record)
        elif operation == 'chat-config':
          if session_rules.chat_config_locked(record):
            raise ValueError(
                'Captured Chat configuration is immutable; create a new Chat'
            )
          record.update(
              generation=generation_config.generation_config(
                  payload.get('generation')
              ),
              initialized=False,
              status='draft',
          )
        elif operation == 'update':
          if record.get('has_capture'):
            raise ValueError(
                'Captured configuration is immutable. Duplicate it to edit a'
                ' new configuration.'
            )
          record.update(self._config(payload))
        elif operation == 'rename':
          record['name'] = self.validate_name(payload.get('name'))
        else:
          raise ValueError('Unknown session operation.')
      self._write_workspace(workspace)
      return record

  def rename(self, payload):
    if isinstance(payload, dict) and payload.get('id') != self.identity:
      return self.manage('rename', payload)
    with self.lock:
      if not isinstance(payload, dict) or payload.get('id') != self.identity:
        raise ValueError(
            'Session identity does not match the configured capture.'
        )
      self._find(self._workspace(), self.identity)
      name = self.validate_name(payload.get('name'))
      # Write only UI metadata, atomically, next to the capture.
      fsutil.atomic_json(self.path, {'id': self.identity, 'name': name})
      return self.summary()
