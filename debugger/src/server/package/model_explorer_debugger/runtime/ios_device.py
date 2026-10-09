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

"""Persistent WebSocket control over a USB-only iproxy tunnel.

The browser continues to use the existing HTTP/SSE jobs API. Device files use
CoreDevice app-container transfer; commands and token deltas use WebSocket.
"""

import json
import os
import pathlib
import re
import secrets
import shutil
import socket
import subprocess
import tempfile
import time

from model_explorer_debugger.runtime import protocol
from model_explorer_debugger.runtime import runner_artifacts
from model_explorer_debugger.runtime import runner_channel
from model_explorer_debugger.runtime import runner_device
from websockets import exceptions as ws_exceptions
from websockets import protocol as ws_protocol
from websockets.sync import client as ws_client

BUNDLE = 'dev.modeldebugger.runner'
DEVICE_PORT = 8769


def device_id(value):
  """Extracts and validates the raw UDID from an ios:<UDID> device ID."""
  if not isinstance(value, str) or not re.fullmatch(
      r'ios:[A-Za-z0-9-]{16,64}', value
  ):
    raise ValueError('Choose an iPhone from /api/devices (device: ios:<UDID>)')
  return value[4:]


def devicectl(*arguments, timeout=30, cancelled=None):
  """Runs xcrun devicectl with JSON output and cancellation polling."""
  with tempfile.TemporaryDirectory(prefix='debugger-device-') as directory:
    output = pathlib.Path(directory) / 'result.json'
    with subprocess.Popen(
        [
            'xcrun',
            'devicectl',
            '--json-output',
            str(output),
            '--timeout',
            str(timeout),
            *map(str, arguments),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
      deadline = time.monotonic() + timeout + 5
      while True:
        try:
          stdout, stderr = process.communicate(timeout=0.1)
          break
        except subprocess.TimeoutExpired:
          interrupted = cancelled is not None and cancelled()
          if interrupted or time.monotonic() > deadline:
            process.kill()
            process.communicate()
            if interrupted:
              raise InterruptedError('Cancelled USB file transfer')
            raise TimeoutError('Device command timed out')
    if process.returncode:
      raise RuntimeError((stderr or stdout)[-3000:])
    data = json.loads(output.read_text())
    if data['info']['outcome'] != 'success':
      raise RuntimeError('Device command failed')
    return data['result']


def list_devices():
  """Lists wired USB iOS devices discovered via xcrun devicectl."""
  if not shutil.which('xcrun'):
    return []
  result = devicectl('list', 'devices')
  return [
      dict(
          id='ios:' + d['hardwareProperties']['udid'],
          name=d.get('deviceProperties', {}).get('name', 'iPhone'),
          platform='iOS',
          transport='USB',
          connection=d.get('connectionProperties', {}).get(
              'transportType', 'unknown'
          ),
      )
      for d in result.get('devices', [])
      if d.get('hardwareProperties', {}).get('platform') == 'iOS'
      and d.get('connectionProperties', {}).get('transportType') == 'wired'
      and d.get('hardwareProperties', {}).get('udid')
  ]


def validate_request(request):
  """Validates an iOS Runner execution request."""
  device_id(request['run'].get('device'))
  runner_device.validate_native_request(request)


class USBConnection(
    runner_artifacts.RunnerArtifacts, runner_channel.RunnerChannel
):
  """Authenticated WebSocket channel tunneled over USB iproxy to an iPhone."""

  @property
  def is_open(self):
    return (
        self.socket is not None and self.socket.state is ws_protocol.State.OPEN
    )

  def __init__(self, udid, directory):
    super().__init__()
    iproxy = shutil.which('iproxy')
    if not iproxy:
      raise ValueError(
          'Install the USB transport on this Mac: brew install libusbmuxd'
      )
    self.udid = udid
    self.socket = None
    self.proxy = None
    self.log = None
    inspection = directory / 'inspection.json'
    try:
      token = json.loads(inspection.read_text())['token']
      if not re.fullmatch('[0-9a-f]{64}', token):
        raise ValueError('Invalid saved USB pairing token')
    except FileNotFoundError:
      token = secrets.token_hex(32)
    try:
      # Provision via the already trusted app container; never put the
      # bearer token in process arguments, API results or event logs.
      with tempfile.TemporaryDirectory(prefix='debugger-usb-') as temporary:
        config = pathlib.Path(temporary) / 'usb-bridge.json'
        config.write_text(json.dumps({'token': token}))
        config.chmod(0o600)
        self.copy_to(config, 'Documents/usb-bridge.json')
      devicectl(
          'device',
          'process',
          'launch',
          '--device',
          udid,
          BUNDLE,
          '--runner-session',
      )
      with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
      directory.mkdir(parents=True, exist_ok=True)
      self.log = (directory / 'iproxy.log').open('ab')
      self.proxy = subprocess.Popen(
          [
              iproxy,
              '--local',
              '--udid',
              udid,
              '--source',
              '127.0.0.1',
              f'{port}:{DEVICE_PORT}',
          ],
          stdout=self.log,
          stderr=self.log,
      )
      deadline = time.monotonic() + 20
      while True:
        try:
          if self.proxy.poll() is not None:
            raise RuntimeError('USB proxy exited; see iproxy.log')
          self.socket = ws_client.connect(
              f'ws://127.0.0.1:{port}',
              additional_headers={'Authorization': 'Bearer ' + token},
              proxy=None,
              compression=None,
              open_timeout=2,
              close_timeout=1,
              ping_interval=None,
              max_size=1024 * 1024,
          )
          hello = self.receive(5)
          if (
              hello.get('type') != 'hello'
              or hello.get('protocolVersion')
              not in protocol.SUPPORTED_PROTOCOL_VERSIONS
              or not hello.get('debuggerEnabled')
          ):
            raise ValueError(
                'Unsupported iPhone Runner protocol or debugger build'
            )
          # A later read-only inspection can reuse this pairing without
          # launching the App or touching the active control tunnel.
          inspection = directory / 'inspection.json'
          descriptor = os.open(
              inspection, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
          )
          with os.fdopen(descriptor, 'w') as saved:
            json.dump({'token': token}, saved)
          inspection.chmod(0o600)
          self.start_channel(hello)
          break
        except (OSError, TimeoutError, ws_exceptions.InvalidHandshake):
          if self.socket:
            self.socket.close()
            self.socket = None
          if time.monotonic() >= deadline:
            raise RuntimeError(
                'USB WebSocket connection failed. Unlock the iPhone and open'
                ' the updated Runner.'
            )
          time.sleep(0.25)
    except BaseException:
      self.close()
      raise

  def copy_to(self, source, destination, cancelled=None):
    return devicectl(
        'device',
        'copy',
        'to',
        '--device',
        self.udid,
        '--source',
        source,
        '--destination',
        destination,
        '--domain-type',
        'appDataContainer',
        '--domain-identifier',
        BUNDLE,
        timeout=600,
        cancelled=cancelled,
    )

  def ensure_model(self, model, sha, emit, cancelled=None):
    reply = runner_channel.call(
        self,
        'model_info',
        timeout=10,
        failure='Unexpected model-cache response',
        modelSHA256=sha,
    )
    if not reply.get('available'):
      emit('progress', message='Transferring model to iPhone over USB…')
      self.copy_to(
          model, f'Documents/Models/{sha}.pending', cancelled=cancelled
      )
      reply = runner_channel.call(
          self,
          'model_commit',
          timeout=10,
          reply='model_info',
          failure='iPhone could not commit the transferred model',
          modelSHA256=sha,
      )
      if not reply.get('available'):
        raise ValueError('iPhone could not commit the transferred model')
    else:
      emit(
          'progress',
          message=(
              'Using model cached on iPhone; its hash will be verified before'
              ' execution'
          ),
      )

  def close(self):
    self.stop_channel()
    socket, proxy, log = self.socket, self.proxy, self.log
    self.socket = self.proxy = self.log = None
    try:
      if socket:
        socket.close()
    finally:
      try:
        if proxy:
          proxy.terminate()
          try:
            proxy.wait(timeout=3)
          except subprocess.TimeoutExpired:
            proxy.kill()
            proxy.wait()
      finally:
        if log:
          log.close()


class IOSDevices(runner_device.RunnerDevices):
  DEVICE_PREFIXES = ('ios:',)
  SUPPORTED_RUNTIMES = frozenset({'LiteRT-LM'})
  SUPPORTS_MULTI_SLOT = False
  platform = 'iOS'
  transport = 'USB WebSocket'
  capture_folder = 'phone-capture'
  device_id = staticmethod(device_id)
  validate = staticmethod(validate_request)

  def connect(self, udid):
    return USBConnection(udid, self.root / 'ios-devices' / udid)
