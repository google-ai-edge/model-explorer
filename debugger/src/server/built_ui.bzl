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

"""The built UI, as a repository the wheel takes it from.

The UI is built by npm in src/ui, outside Bazel. This repository rule makes
the directory npm writes, if built, a repository whose target :ui holds its
files under model_debugger/server/ui/, where the wheel carries them. It is
fetched anew whenever that directory changes or appears. If the UI is not
built, building :ui fails, saying how to build it; nothing else depends on
it.
"""

_FILES_BUILD = """\
filegroup(
    name = "ui",
    srcs = glob(["model_debugger/server/ui/**"]),
    visibility = ["//visibility:public"],
)
"""

_MISSING_BZL = """\
def _missing_impl(ctx):
    fail(ctx.attr.message)

missing = rule(
    implementation = _missing_impl,
    attrs = {"message": attr.string()},
)
"""

_MISSING_BUILD = """\
load(":missing.bzl", "missing")

missing(
    name = "ui",
    message = {message},
    visibility = ["//visibility:public"],
)
"""

def _built_ui_impl(rctx):
    ui = rctx.path("{}/{}".format(rctx.workspace_root, rctx.attr.path))
    index = ui.get_child("index.html")

    # A change to the file's existence counts too.
    rctx.watch(index)
    if index.exists:
        rctx.watch_tree(ui)
        rctx.symlink(ui, "model_debugger/server/ui")
        rctx.file("BUILD.bazel", _FILES_BUILD)
        return
    rctx.file("missing.bzl", _MISSING_BZL)
    message = (
        "The wheel carries the built UI, which is not in {}. Build it " +
        "first: cd src/ui && npm ci && npm run build"
    ).format(ui)
    rctx.file("BUILD.bazel", _MISSING_BUILD.format(message = repr(message)))

built_ui = repository_rule(
    implementation = _built_ui_impl,
    attrs = {
        "path": attr.string(
            mandatory = True,
            doc = "The built UI's directory, relative to this module's root.",
        ),
    },
    doc = "The built UI's files, under model_debugger/server/ui/.",
)
