# Copyright 2025 The AI Edge Model Explorer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Custom repository rule for fetching the TensorFlow source code.
This rule allows selecting between a local TensorFlow source directory or a
remote GitHub archive based on environment variables. This provides flexibility
for developers and ensures reproducible builds in CI.
"""

def _tensorflow_source_repo_impl(ctx):
    use_local_tf = ctx.os.environ.get("USE_LOCAL_TF", "false").lower() == "true"

    if use_local_tf:
        # If using a local TF source, the path must be provided.
        local_path_env = ctx.os.environ.get("TF_LOCAL_SOURCE_PATH")
        if not local_path_env:
            fail(
                "ERROR: USE_LOCAL_TF is set to 'true', but the " +
                "TF_LOCAL_SOURCE_PATH environment variable is not set. " +
                "Please set it to the absolute path of your local TensorFlow repository.",
            )

        # Symlink the contents of the local directory into the Bazel external repository.
        # This makes the local source available to the build.
        resolved_local_path = ctx.path(local_path_env)
        if not resolved_local_path.exists:
            fail("ERROR: The path specified by TF_LOCAL_SOURCE_PATH does not exist: %s" % local_path_env)

        ctx.symlink(resolved_local_path, ctx.name)

    else:
        # If not using a local source, download from the specified URL.
        # This is the standard CI/release build path.
        if not ctx.attr.sha256:
            fail("ERROR: The 'sha256' attribute must be set when not using a local TF source.")

        ctx.download_and_extract(
            url = ctx.attr.urls,
            sha256 = ctx.attr.sha256,
            stripPrefix = "tensorflow-" + ctx.attr.commit,
        )

tensorflow_source_repo = repository_rule(
    implementation = _tensorflow_source_repo_impl,
    # This rule is sensitive to environment variables, so we mark it as `local`.
    local = True,
    attrs = {
        "commit": attr.string(mandatory = True, doc = "The git commit hash of TensorFlow."),
        "sha256": attr.string(mandatory = True, doc = "The SHA256 hash of the archive."),
        "urls": attr.string_list(mandatory = True, doc = "URLs to download the TensorFlow archive from."),
    },
    doc = """
    A custom repository rule to select between a local TensorFlow source or a
    remote http_archive based on the 'USE_LOCAL_TF' and 'TF_LOCAL_SOURCE_PATH'
    environment variables.
    """,
)
