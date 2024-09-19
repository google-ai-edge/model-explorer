# Copyright 2024 The AI Edge Model Explorer Authors.
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

import queue


class ServerDirectiveDispatcher:
  """
  A class to register server directive listeners and dispatch messages to them.
  """

  def __init__(self):
    self.listeners = []

  def listen(self):
    q = queue.Queue[str](maxsize=10)
    self.listeners.append(q)
    return q

  def broadcast(self, msg: str):
    for listeners in self.listeners:
      listeners.put_nowait(msg)

  def remove_listener(self, listener: queue.Queue[str]):
    self.listeners.remove(listener)
