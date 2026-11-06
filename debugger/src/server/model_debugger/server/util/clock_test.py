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

"""Tests of the current time."""

import datetime

from absl.testing import absltest
from model_debugger.server.testing import test_main
from model_debugger.server.util import clock


class ClockTest(absltest.TestCase):

  def test_now_is_utc_to_the_microsecond(self):
    before = datetime.datetime.now(datetime.UTC)
    value = clock.now()
    after = datetime.datetime.now(datetime.UTC)
    self.assertRegex(value, r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}\+00:00$')
    self.assertBetween(datetime.datetime.fromisoformat(value), before, after)


if __name__ == '__main__':
  test_main.main()
