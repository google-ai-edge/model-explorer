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

"""The real command-line runner, for the tests that run it when it is built.

RUNNER_BINARY names one; otherwise it is the one Bazel built last, as the
command line finds it.
"""

import os
import pathlib

from model_debugger.server.cli import server_main
from model_debugger.server.runner import local_launcher
from model_debugger.server.runner import runner_builds

# RUNNER_BINARY, else the runner Bazel built last; None when neither is.
RUNNER = (
    pathlib.Path(os.environ['RUNNER_BINARY'])
    if os.environ.get('RUNNER_BINARY')
    else server_main.built_runner()
)
# Whether RUNNER is there to run.
BUILT = RUNNER is not None and RUNNER.is_file()
# The build to launch it as: it is this computer's development build.
BUILD = runner_builds.DEVELOPMENT


def launcher() -> local_launcher.LocalLauncher:
  """This computer, with RUNNER as its development build and no other."""
  return local_launcher.LocalLauncher(runner_builds.RunnerBuilds(None, RUNNER))
