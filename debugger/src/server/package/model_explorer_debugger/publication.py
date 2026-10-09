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

"""Durable text Turns and their capture publication, with restart recovery."""

from copy import deepcopy

from .capture_importer import publish_capture


class Publication:

  def __init__(self, manager):
    self.manager = manager

  def publish_turn(self, job, record, results, artifacts):
    manager, registry = self.manager, self.manager.registry
    text_turn = dict(
        n=job['turn'],
        job_id=job['id'],
        input=job['prompt'],
        output={role: results[role]['output'] for role in ('ref', 'target')},
        generation_status='completed',
        debug_data={'status': 'pending'},
    )
    publication = dict(
        job_id=job['id'],
        session_id=record['id'],
        turn=job['turn'],
        status='pending',
        record=record,
        results=results,
        artifacts=artifacts,
        requests={
            run['id']: manager._request(
                record,
                run,
                artifacts[run['id']],
                manager.directory(job),
                'generate',
                prompt=job['prompt'],
                turn=job['turn'],
            )
            for run in record['runs']
        },
    )
    with manager.lock, registry.transaction():
      saved = deepcopy(registry.get(record['id']).get('successful_turns', []))
      if not any(turn.get('job_id') == job['id'] for turn in saved):
        saved.append(text_turn)
      registry.update(record['id'], successful_turns=saved)
      registry.database.publication(job['id'], publication)
    self.finish(publication)

  def finish(self, publication):
    from .capture_publication import resolve_publication

    manager, registry = self.manager, self.manager.registry
    debug = resolve_publication(
        registry,
        publication,
        getattr(manager.runners, 'import_result', None),
        publish=publish_capture,
    )
    identity = publication['session_id']
    with manager.lock:
      job = manager.jobs.get(publication['job_id'])
      previous = deepcopy(job) if job else None
      try:
        with manager.lock, registry.transaction():
          saved = deepcopy(registry.get(identity).get('successful_turns', []))
          turn = next(
              row for row in saved if row.get('job_id') == publication['job_id']
          )
          turn['debug_data'] = debug
          fields = dict(
              successful_turns=saved,
              notice='Turn saved.'
              if debug['status'] == 'available'
              else 'Turn saved. Debug data is unavailable.',
          )
          if debug['status'] == 'available':
            fields.update(capture=debug['capture'], has_capture=True)
          # Recovery preserves the read-only interrupted execution state.
          if registry.get(identity).get('initialized'):
            fields['status'] = 'saved'
          registry.update(identity, **fields)
          registry.database.publication(
              publication['job_id'],
              {**publication, 'status': 'complete', 'debug_data': debug},
          )
          job = manager.jobs.get(publication['job_id'])
          if job:
            job.update(
                generation_status='completed',
                debug_data=debug,
                turn_published=True,
            )
            manager.record_event(job, 'publication', debug_data=debug)
      except BaseException:
        if job is not None:
          job.clear()
          job.update(previous)
        raise

  def recover(self):
    registry = self.manager.registry
    publications = registry.database.publications()
    for value in publications.values():
      if value['status'] == 'pending':
        record = next(
            (
                row
                for row in registry.state['sessions']
                if row['id'] == value['session_id']
            ),
            None,
        )
        if record is None or not any(
            turn.get('job_id') == value['job_id']
            for turn in record.get('successful_turns', [])
        ):
          registry.database.publication(
              value['job_id'],
              {
                  **value,
                  'status': 'complete',
                  'debug_data': {
                      'status': 'unavailable',
                      'error': 'Publication owner or saved turn is missing',
                  },
              },
          )
          continue
        self.finish(value)
    # Older workspaces may contain pending text with no publication journal.
    # Preserve its text and explicitly resolve missing recovery evidence.
    for record in list(registry.state['sessions']):
      turns = deepcopy(record.get('successful_turns', []))
      changed = False
      for turn in turns:
        if (
            turn.get('debug_data', {}).get('status') == 'pending'
            and turn.get('job_id') not in publications
        ):
          turn['debug_data'] = {
              'status': 'unavailable',
              'error': (
                  'Server stopped before capture recovery evidence was saved.'
              ),
          }
          changed = True
      if changed:
        registry.update(record['id'], successful_turns=turns)
