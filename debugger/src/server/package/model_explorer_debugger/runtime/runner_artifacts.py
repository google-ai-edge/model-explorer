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

"""Immutable dump reception shared by Mac and USB Runner channels."""

import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from uuid import UUID, uuid4

from ..fsutil import clone_file, file_digest
from .protocol import FILE_CHUNK as CHUNK, FILE_WINDOW
from .runner_channel import call, pipeline

# Metadata files a sealed run export may contain. RunnerFiles.swift
# (run_files/safeFile) builds the same list; docs/data-contract.md documents the
# export layout.
RUN_METADATA_FILES = (
    'job.json',
    'result.json',
    'runtime-build.json',
    'terminal.json',
    'failure.json',
)


def _sync_directory(path):
  descriptor = os.open(path, os.O_RDONLY)
  try:
    os.fsync(descriptor)
  finally:
    os.close(descriptor)


class RunnerArtifacts:

  def exchange(self, kind, **fields):
    return call(self, kind, **fields)

  @staticmethod
  def _shared_export(root, run):
    """The Runner's export directory, or None.

    Returned only when the Runner offered one and it really is this run's.
    """
    if not isinstance(root, str):
      return None
    directory = Path(root)
    if (
        not directory.is_absolute()
        or directory.name.lower() != run
        or directory.parent.name != 'Runs'
        or not directory.is_dir()
    ):
      raise ValueError('Runner offered an invalid shared export directory')
    return directory

  def _chunks(self, run, manifest_id, path, size):
    """Replies for every chunk of one sealed file, in offset order."""
    request = dict(run=run, manifestId=manifest_id, path=str(path))
    if getattr(self, 'protocol', 4) >= 5:
      return pipeline(
          self,
          (
              (
                  'file_chunk',
                  dict(request, offset=offset, encoding='binary'),
                  None,
              )
              for offset in range(0, size, CHUNK)
          ),
          window=FILE_WINDOW,
      )
    return (
        self.exchange('file_chunk', **request, offset=offset)
        for offset in range(0, size, CHUNK)
    )

  def copy_from(self, source, destination, cancelled=None):
    """Stage, verify and fsync an entire manifest before acknowledging it.

    Generation cancellation deliberately does not cancel terminal dump
    reception. Connection loss ends the operation; no reconnect or receipt
    retry is scheduled. Receipt failure never invalidates durable files.
    """
    prefix = 'Documents/Runs/'
    if not source.startswith(prefix):
      raise ValueError('Invalid capture directory')
    run = str(UUID(source[len(prefix) :]))
    # A Runner sharing this filesystem names its export directory instead of
    # streaming it.
    manifest = self.exchange(
        'run_files',
        run=run,
        **({'link': True} if getattr(self, 'links_files', False) else {}),
    )
    shared = self._shared_export(manifest.get('root'), run)
    manifest_id = str(UUID(manifest['manifestId']))
    files = manifest.get('files')
    if not isinstance(files, list) or not 1 <= len(files) <= 2048:
      raise ValueError('Invalid capture file list')
    status = manifest.get('generationStatus')
    if status not in ('succeeded', 'stopped', 'failed'):
      raise ValueError('Invalid dump terminal status')
    seen, records, total = set(), [], 0
    for record in files:
      path = PurePosixPath(record['path'])
      valid = (
          str(path) in RUN_METADATA_FILES
          or len(path.parts) >= 2
          and path.parts[0] in ('tensors', 'raw')
      )
      if (
          not valid
          or path.is_absolute()
          or any(p in ('..', '.') for p in path.parts)
          or str(path) in seen
          or str(path) != record['path']
      ):
        raise ValueError('Unsafe or duplicate capture path')
      seen.add(str(path))
      size, checksum = record['size'], record['sha256']
      if (
          type(size) is not int
          or not 0 <= size <= 4 * 1024**3
          or not isinstance(checksum, str)
          or not re.fullmatch('[0-9a-f]{64}', checksum)
      ):
        raise ValueError('Invalid capture size or checksum')
      total += size
      if total > 8 * 1024**3:
        raise ValueError('Capture exceeds transfer limit')
      records.append((path, size, checksum))
    if 'terminal.json' not in seen:
      raise ValueError('Missing dump terminal metadata')
    destination = Path(destination)
    staging = destination.with_name(
        destination.name + '.receiving-' + str(uuid4())
    )
    staging.mkdir(parents=True)
    try:
      for path, size, checksum in records:
        target = staging / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if shared:
          clone_file(shared / path, target)
          if target.stat().st_size != size or file_digest(target) != checksum:
            raise ValueError('Captured file checksum mismatch')
          with target.open('rb') as output:
            os.fsync(output.fileno())
          continue
        digest = hashlib.sha256()
        replies = self._chunks(run, manifest_id, path, size)
        try:
          with target.open('xb') as output:
            for offset, reply in zip(range(0, size, CHUNK), replies):
              data = (
                  reply['data']
                  if isinstance(reply['data'], bytes)
                  else base64.b64decode(reply['data'], validate=True)
              )
              if reply.get('offset') != offset or len(data) != min(
                  CHUNK, size - offset
              ):
                raise ValueError('Invalid capture chunk')
              output.write(data)
              digest.update(data)
            output.flush()
            os.fsync(output.fileno())
        finally:
          replies.close()
        if digest.hexdigest() != checksum:
          raise ValueError('Captured file checksum mismatch')
      terminal = json.loads((staging / 'terminal.json').read_text())
      if terminal.get('generationStatus') != status:
        raise ValueError('Dump manifest and terminal status differ')
      terminal_run = terminal.get(
          'jobID', terminal.get('requestId', terminal.get('run'))
      )
      if terminal_run is not None and str(UUID(terminal_run)) != run:
        raise ValueError('Dump terminal identity mismatch')
      with (staging / 'received-manifest.json').open('x') as saved:
        json.dump(manifest, saved)
        saved.flush()
        os.fsync(saved.fileno())
      for directory in sorted(
          (p for p in staging.rglob('*') if p.is_dir()),
          key=lambda p: len(p.parts),
          reverse=True,
      ):
        _sync_directory(directory)
      _sync_directory(staging)
      if destination.exists():
        raise ValueError('Capture destination already exists')
      staging.rename(destination)
      _sync_directory(destination.parent)
    except BaseException:
      shutil.rmtree(staging, ignore_errors=True)
      raise
    receipt = dict(
        dump_complete=True,
        manifest_id=manifest_id,
        dump_generation_status=status,
        cleanup_error='',
        connection_lost=False,
    )
    if not getattr(self, 'is_open', True):
      receipt['connection_lost'] = True
      return receipt
    try:
      acknowledged = self.exchange(
          'run_received', run=run, manifestId=manifest_id
      )
      if acknowledged.get('deleted') is not True:
        receipt['cleanup_error'] = 'Runner did not confirm dump deletion'
    except Exception as error:
      receipt['connection_lost'] = not getattr(
          self, 'is_open', True
      ) or isinstance(error, (ConnectionError, TimeoutError))
      if not receipt['connection_lost']:
        receipt['cleanup_error'] = str(error)
    return receipt
