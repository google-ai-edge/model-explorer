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

"""Tests that load a model: once on the CPU, once on the GPU."""

load("@rules_cc//cc:defs.bzl", "cc_test")

def model_test(name, srcs, deps):
    """Declares `name`, the test on the CPU, and `name`_gpu, the same on the GPU.

    Both read the model from RUNNER_MODEL (.bazelrc passes it through) and skip
    without one. The GPU test is told its backend; the runtime finds its
    plugins beside its own library.

    Args:
      name: The CPU test's name.
      srcs: The test's sources.
      deps: The test's dependencies.
    """
    cc_test(
        name = name,
        srcs = srcs,
        deps = deps,
    )
    cc_test(
        name = name + "_gpu",
        srcs = srcs,
        env = {"RUNNER_BACKEND": "gpu"},
        deps = deps,
    )
