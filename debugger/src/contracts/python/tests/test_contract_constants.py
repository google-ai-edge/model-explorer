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

"""Wire constants both Server and Runner must agree on."""

import unittest

from model_debugger_contracts.errors import InputRejected
from model_debugger_contracts.jobs import TERMINAL_STATUSES


class ContractConstantsTests(unittest.TestCase):

  def test_input_rejected_code_is_the_wire_error_code(self):
    self.assertEqual(InputRejected.code, 'input_rejected')
    self.assertEqual(InputRejected('too long').code, 'input_rejected')

  def test_terminal_statuses_are_fixed(self):
    self.assertEqual(TERMINAL_STATUSES, ('completed', 'failed', 'cancelled'))


if __name__ == '__main__':
  unittest.main()
