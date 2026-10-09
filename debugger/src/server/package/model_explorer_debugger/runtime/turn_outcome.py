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

"""One side's turn result along its three independent axes.

Text, Debug data and the link fail independently: confirmed text survives a
failed dump, and a failed dump does not by itself mean the Runner is gone. These
helpers are the only writers of the flat fields that jobs.py, publication and
the UI read.
"""


class RunnerRefused(RuntimeError):
  """The Runner answered with a failure.

  The operation is over and the link is still in step.
  """


def text(status, output=''):
  """`status` is completed, stopped or failed.

  Only completed text is ever reported.
  """
  confirmed = status == 'completed' and isinstance(output, str)
  return dict(
      output=output if confirmed else '',
      output_confirmed=confirmed,
      generation_status=status,
  )


def debug(error=None):
  unavailable = error is not None
  return {
      'debug_data': {
          'status': 'unavailable' if unavailable else 'available',
          'error': str(error or ''),
      }
  }


def dump_failed(error, *, lost):
  """The dump, or its receipt, failed after the text was settled."""
  return dict(debug(error), dump_error=str(error), connection_lost=lost)


def unconfirmed(error, **fields):
  """The link failed before any text was confirmed.

  The prompt is never replayed.
  """
  return dict(
      fields,
      **text('failed'),
      **debug(error),
      dump_complete=False,
      connection_lost=True,
      error=str(error),
  )
