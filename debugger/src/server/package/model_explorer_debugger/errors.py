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

"""Client-facing request errors and the HTTP status each one maps to.

`ValueError` is the domain contract for a request the client must change: the
transport reports it as HTTP 400 with its message. `NotFound` is HTTP 404 and
`AnalysisBusy` is HTTP 503. Every other exception is a server fault, reported
as a generic HTTP 500 and logged.

`RequestFields` wraps untrusted query parameters and JSON objects so that a
missing field is reported as a client error.
"""

from typing import Any


class BadRequestError(ValueError):
  """The request is malformed; reported as HTTP 400."""


BadRequest = BadRequestError


class NotFoundError(ValueError):
  """A route or a client-named resource does not exist; reported as HTTP 404."""


NotFound = NotFoundError


class MissingFieldError(BadRequestError, KeyError):
  """A required query parameter or JSON object field is absent.

  It is also a `KeyError`, so code that handles a missing key keeps working.
  """

  def __str__(self) -> str:
    return f'missing_field: {self.args[0]}'


MissingField = MissingFieldError


class UnknownResourceError(NotFoundError, KeyError):
  """A capture resource or batch ID is not in the selected capture.

  It is also a `KeyError`, so code that handles a missing key keeps working.
  """

  def __str__(self) -> str:
    return str(self.args[0])


UnknownResource = UnknownResourceError


class AnalysisBusyError(RuntimeError):
  """Analysis capacity is saturated, cancelled, or timed out; HTTP 503."""


AnalysisBusy = AnalysisBusyError


class RequestFields(dict[str, Any]):
  """Query parameters or a JSON object received from a client.

  Indexing a missing key raises `MissingField` instead of a bare `KeyError`, so
  an absent field is reported as a client error rather than a server fault.
  """

  def __missing__(self, key: str) -> Any:
    raise MissingFieldError(key)
