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

"""The current time, as every record and view of this server writes it."""

import datetime


def now() -> str:
  """Returns the current UTC time in ISO 8601.

  Returns:
      The time to the microsecond with its offset, such as
      `2026-10-15T09:30:00.000000+00:00`.
  """
  return datetime.datetime.now(datetime.UTC).isoformat(timespec='microseconds')
