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

"""The two refusals that are not a bad request.

A `ValueError` says the request itself is wrong; these say it is right but
cannot be done.
"""


class NotFoundError(Exception):
  """What the request names does not exist."""


class ConflictError(Exception):
  """The request is valid but the server's state refuses it now."""


# The old names, still used by runner/, service/ and session/.
NotFound = NotFoundError
Conflict = ConflictError


def message(error: Exception) -> str:
  """Returns what the user is told of a refusal.

  Args:
      error: A NotFoundError, a ConflictError, or a ValueError, KeyError or
        TypeError of a bad request.

  Returns:
      Its text; a KeyError's str() is the key in quotes, so its argument
      is named instead.
  """
  if isinstance(error, KeyError) and error.args:
    return f'Missing {error.args[0]}.'
  return str(error)
