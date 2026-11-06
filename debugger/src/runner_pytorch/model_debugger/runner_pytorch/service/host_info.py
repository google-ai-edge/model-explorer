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

"""The machine the runner runs on, as its system reports it, and its GPU.

Nothing is guessed: what the system does not report is left unset. The GPU
is the one PyTorch can run on: MPS on a Mac, CUDA elsewhere.
"""

import os
import pathlib
import platform
import subprocess

from model_debugger.protocol import runner_pb2
import torch

_PLATFORMS = {
    'Darwin': runner_pb2.HostInfo.MACOS,
    'Linux': runner_pb2.HostInfo.LINUX,
}
_ARCHITECTURES = {'aarch64': 'arm64', 'amd64': 'x86_64'}


def gpu_device() -> str | None:
  """Returns the torch device of the GPU PyTorch can use, or None."""
  if torch.backends.mps.is_available():
    return 'mps'
  if torch.cuda.is_available():
    return 'cuda'
  return None


def measure_host() -> runner_pb2.HostInfo:
  """Returns this machine as its system reports it."""
  system = platform.system()
  host = runner_pb2.HostInfo(
      platform=_PLATFORMS.get(system, runner_pb2.HostInfo.PLATFORM_UNSPECIFIED),
      architecture=_ARCHITECTURES.get(platform.machine(), platform.machine()),
      cpu_count=os.cpu_count() or 0,
  )
  if system == 'Darwin':
    host.os_version = _run('sw_vers', '-productVersion')
    host.os_build = _run('sw_vers', '-buildVersion')
    host.model = _run('sysctl', '-n', 'hw.model')
    chip = _run('sysctl', '-n', 'machdep.cpu.brand_string')
    memory = _run('sysctl', '-n', 'hw.memsize')
  else:
    host.os_version = platform.release()
    host.os_build = platform.version()
    host.model = _read('/sys/devices/virtual/dmi/id/product_name')
    chip = _cpu_model()
    memory = str(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
  if chip:
    host.chip = chip
  if memory.isdigit():
    host.memory_bytes = int(memory)
  device = gpu_device()
  if device == 'mps':
    # Apple's GPU is part of the chip, and its memory unified.
    host.gpu.name = chip
  elif device == 'cuda':
    properties = torch.cuda.get_device_properties(0)
    host.gpu.name = properties.name
    host.gpu.memory_bytes = properties.total_memory
  return host


def _run(*command: str) -> str:
  """Returns what a command prints, stripped; empty if it fails."""
  try:
    return subprocess.run(
        command, capture_output=True, text=True, check=True
    ).stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return ''


def _read(path: str) -> str:
  """Returns a file's text, stripped; empty if it cannot be read."""
  try:
    return pathlib.Path(path).read_text().strip()
  except OSError:
    return ''


def _cpu_model() -> str:
  """Returns the CPU's model name from /proc/cpuinfo; empty if none."""
  for line in _read('/proc/cpuinfo').splitlines():
    name, _, value = line.partition(':')
    if name.strip() == 'model name':
      return value.strip()
  return ''
