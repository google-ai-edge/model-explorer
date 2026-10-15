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

"""The errors the HTTP layer answers with a status of their own.

Any other `ValueError` is a refused request (400); these two are not.
"""


class NotFound(Exception):
  """What the request names does not exist: answered with 404."""


class Conflict(Exception):
  """The request is valid but the server's state refuses it now: 409."""
