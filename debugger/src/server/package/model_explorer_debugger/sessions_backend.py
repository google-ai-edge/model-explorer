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

"""What the HTTP layer needs from a Session source."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class SessionsBackend(Protocol):
  """Implemented by SessionMetadata and SessionRegistry.

  SessionMetadata backs one saved capture (`--data`) and SessionRegistry backs a
  workspace (`--workspace`). Both apply the rules in session_rules.py; the
  registry additionally owns execution.
  """

  def listing(self) -> dict:
    ...

  def manage(self, operation: str, payload: dict) -> dict:
    ...

  def rename(self, payload: dict) -> dict:
    ...

  def upload(self, stream, size: int, name: str) -> dict:
    ...
