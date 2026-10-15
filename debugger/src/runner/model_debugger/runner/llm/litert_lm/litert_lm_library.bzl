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

"""LiteRT-LM's library, built from a commit of its source by its own Bazel.

LiteRT-LM is a WORKSPACE project on Bazel 7 with TensorFlow's dependencies;
the Runner is a module on Bazel 8 with gRPC's. Neither can host the other, so
the runtime is built apart and joins the Runner as a library that exports its
C API alone: its Abseil and protobuf stay inside it and never meet the Runner's.

The repository is a function of the commit and the platform: fetched and
built once, kept by Bazel's cache until either changes. LiteRT-LM's own
Bazel keeps its outputs in a directory of the user's cache named after the
two, so that a repository fetched again (its rule or templates edited, the
Runner's checkout moved) links what is built rather than compiling LiteRT-LM
again. The GPU plugins are what LiteRT-LM ships prebuilt (Git LFS), taken as
they are. Apple platforms get Mach-O libraries linking LiteRT dynamically;
Android an ELF library with LiteRT linked in, built with the NDK that
ANDROID_NDK_HOME names.
"""

# The lowest macOS and iOS the library is built for, and so the lowest the
# Runner and its Apps run on: LiteRT-LM's prebuilt
# libGemmaModelConstraintProvider, which the runtime links, needs iOS 26.4, and
# macOS follows so that the Apple platforms share one number.
APPLE_MINIMUM_OS = "26.4"

# Per platform:
#   flags       what LiteRT-LM is built with: its own .bazelrc config for the
#               platform, and the Runner's minimum OS after it.
#   elf         an ELF library (Android) rather than Mach-O (Apple).
#   linked      what the runtime links from its own directory. Apple links
#               LiteRT dynamically, as LiteRT-LM ships it; Android has no
#               libLiteRt.so and links it in.
#   linkopts    what the runtime's link is given besides. LiteRT builds its
#               libLiteRt as a macOS library on every Apple platform, and the
#               iOS linker skips it with a warning; there LiteRT-LM's prebuilt
#               libLiteRt, the one an App bundles, is linked by its path in
#               the execution root, where the source's directories appear.
#   plugins     the GPU plugins the runtime loads at run time, in load order.
#   frameworks  whether an App bundles the libraries as frameworks (iOS).
_PLATFORMS = {
    "macos_arm64": struct(
        flags = ["--macos_minimum_os=" + APPLE_MINIMUM_OS],
        elf = False,
        linked = ["libLiteRt", "libGemmaModelConstraintProvider"],
        linkopts = [],
        plugins = ["libwebgpu_dawn", "libLiteRtWebGpuAccelerator", "libLiteRtTopKWebGpuSampler"],
        frameworks = False,
    ),
    "ios_sim_arm64": struct(
        flags = ["--config=ios_sim_arm64", "--ios_minimum_os=" + APPLE_MINIMUM_OS],
        elf = False,
        linked = ["libLiteRt", "libGemmaModelConstraintProvider"],
        linkopts = ["--linkopt=prebuilt/ios_sim_arm64/libLiteRt.dylib"],
        plugins = ["libLiteRtMetalAccelerator", "libLiteRtTopKMetalSampler"],
        frameworks = True,
    ),
    "ios_arm64": struct(
        flags = ["--config=ios_arm64", "--ios_minimum_os=" + APPLE_MINIMUM_OS],
        elf = False,
        linked = ["libLiteRt", "libGemmaModelConstraintProvider"],
        linkopts = ["--linkopt=prebuilt/ios_arm64/libLiteRt.dylib"],
        plugins = ["libLiteRtMetalAccelerator", "libLiteRtTopKMetalSampler"],
        frameworks = True,
    ),
    "android_arm64": struct(
        flags = ["--config=android_arm64"],
        elf = True,
        linked = ["libGemmaModelConstraintProvider"],
        linkopts = [],
        plugins = ["libLiteRtOpenClAccelerator", "libLiteRtTopKOpenClSampler"],
        frameworks = False,
    ),
}

# An iOS App bundles a dynamic library only as a framework: a directory named
# after it, holding the library under that name and a plist.
_FRAMEWORK_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>{name}</string>
  <key>CFBundleIdentifier</key>
  <string>com.google.ai.edge.modeldebugger.runner.{name}</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>{name}</string>
  <key>CFBundlePackageType</key>
  <string>FMWK</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>MinimumOSVersion</key>
  <string>{minimum_os}</string>
</dict>
</plist>
"""

# The C API headers the Runner includes, from LiteRT-LM's c/ directory.
_HEADERS = ["conversation.h", "engine.h", "experimental.h", "model_info.h"]

def _wrap_in_framework(ctx, name, linked, keep_id = False):
    """Turns <name>.dylib in the repository into <name>.framework.

    The library's install name becomes @rpath/<name>.framework/<name>, unless
    `keep_id`, and so does each of `linked` it refers to; then it is signed
    again.
    """
    binary = "{}.framework/{}".format(name, name)

    # The plist first: writing it makes the directory.
    ctx.file("{}.framework/Info.plist".format(name), _FRAMEWORK_PLIST.replace("{name}", name).replace("{minimum_os}", APPLE_MINIMUM_OS))
    _run(ctx, ["mv", name + ".dylib", binary], ".", "mv")
    if not keep_id:
        _run(
            ctx,
            ["install_name_tool", "-id", "@rpath/" + binary, binary],
            ".",
            "install_name_tool -id",
        )
    for other in linked:
        _run(
            ctx,
            [
                "install_name_tool",
                "-change",
                "@rpath/{}.dylib".format(other),
                "@rpath/{}.framework/{}".format(other, other),
                binary,
            ],
            ".",
            "install_name_tool -change",
        )
    _run(ctx, ["codesign", "--force", "--sign", "-", binary], ".", "codesign")

def _run(ctx, arguments, cwd, what, timeout = 600, quiet = True):
    """Runs a command in cwd and fails with its output if it does not succeed."""
    result = ctx.execute(
        arguments,
        working_directory = cwd,
        timeout = timeout,
        environment = ctx.os.environ,
        quiet = quiet,
    )
    if result.return_code != 0:
        fail("litert_lm: {} failed:\n{}{}".format(what, result.stdout, result.stderr))
    return result.stdout

def _fetch_source(ctx, platform):
    """Checks the commit out under source/ and returns that path.

    One commit, no history; then the prebuilt libraries of the platform, which
    are Git LFS pointers until pulled (the C API library links two of them).
    """
    source = ctx.path("source")
    _run(ctx, ["git", "init", "-q", str(source)], ".", "git init")
    _run(ctx, ["git", "remote", "add", "origin", ctx.attr.url], str(source), "git remote add")
    _run(
        ctx,
        ["git", "fetch", "-q", "--depth=1", "origin", ctx.attr.commit],
        str(source),
        "git fetch (the LiteRT-LM commit)",
        timeout = 3600,
    )
    _run(ctx, ["git", "checkout", "-q", "FETCH_HEAD"], str(source), "git checkout")
    _run(ctx, ["git", "lfs", "install", "--local"], str(source), "git lfs install")
    _run(
        ctx,
        ["git", "lfs", "pull", "--include=prebuilt/{}/*".format(platform)],
        str(source),
        "git lfs pull",
        timeout = 3600,
    )
    return source

def _output_base(ctx):
    """Returns the output base of LiteRT-LM's Bazel for this commit and platform.

    It lies in the user's cache, outside every repository: LiteRT-LM's Bazel
    would otherwise key its outputs by the path of the source, which changes
    whenever this repository is fetched somewhere else.
    """
    home = ctx.os.environ.get("HOME")
    if not home:
        fail("litert_lm: HOME is not set; LiteRT-LM is built under ~/Library/Caches")
    return "{}/Library/Caches/model_debugger/litert_lm/{}/{}".format(
        home,
        ctx.attr.commit,
        ctx.attr.platform,
    )

def _build_library(ctx, bazel, source, platform):
    """Builds the C API library with the debugger on and returns its path.

    LiteRT-LM's Bazel server is shut down afterwards, whether or not the build
    succeeded: nothing else uses it, and it would hold gigabytes of memory.
    """
    startup = [bazel, "--output_base=" + _output_base(ctx)]

    # --define turns the debugger on in runtime/core, whose BUILD selects on it;
    # --copt defines the same macro in every file, so that c/experimental.cc, a
    # direct source of //c:litert-lm without a defines attribute, reports it too.
    # Only the C API is exported: were Abseil and protobuf exported too, the
    # Runner's own copies would be coalesced with them at load time and the
    # first destructor would crash.
    if platform.elf:
        script = source.get_child("litert_lm_exports.lds")
        ctx.file(script, "{ global: litert_lm_*; local: *; };\n")
        link = [
            "--linkopt=-Wl,--version-script=" + str(script),
            # The section bounds the linker defines, such as Abseil's
            # __start_malloc_hook, escape the version script; they would meet
            # the Runner's own.
            "--linkopt=-Wl,-z,start-stop-visibility=hidden",
            "--linkopt=-Wl,-soname,liblitert-lm.so",
        ]
    else:
        link = [
            "--define=litert_runtime_link_mode=dynamic",
            "--linkopt=-Wl,-exported_symbol,_litert_lm_*",
        ]

    # A long time the first time.
    built = ctx.execute(
        startup + ["build"] + platform.flags + [
            "--define=LITERT_LM_DEBUGGER_ENABLED=1",
            "--copt=-DLITERT_LM_DEBUGGER_ENABLED=1",
        ] + link + platform.linkopts + ["//c:litert-lm"],
        working_directory = str(source),
        timeout = 4 * 3600,
        environment = ctx.os.environ,
        quiet = False,
    )
    bazel_bin = None
    if built.return_code == 0:
        bazel_bin = ctx.execute(
            startup + ["info", "bazel-bin"] + platform.flags,
            working_directory = str(source),
            environment = ctx.os.environ,
        )
    shutdown = ctx.execute(
        startup + ["shutdown"],
        working_directory = str(source),
        environment = ctx.os.environ,
    )
    if built.return_code != 0:
        fail("litert_lm: building //c:litert-lm failed:\n{}{}".format(built.stdout, built.stderr))
    if bazel_bin.return_code != 0:
        fail("litert_lm: bazel info failed:\n{}{}".format(bazel_bin.stdout, bazel_bin.stderr))
    if shutdown.return_code != 0:
        fail("litert_lm: bazel shutdown failed:\n{}{}".format(shutdown.stdout, shutdown.stderr))
    return bazel_bin.stdout.strip() + "/c/liblitert-lm." + ("so" if platform.elf else "dylib")

def _check_library(ctx, library, platform):
    """Refuses a library without the debugger, or exporting or linking amiss.

    It exports the C API alone. A Mach-O library also records each of
    `linked` as a load command and looks no symbol up at run time: LiteRT-LM's
    .bazelrc links every Apple library with -undefined dynamic_lookup, and a
    library left out of the link would be missed only by the first call.
    """
    elf = platform.elf

    # The debugger is what a Runner is for. Only a debugger build names the
    # directory it captures into; the ELF build strips every symbol, so the
    # text is what survives on every platform.
    if ctx.execute(["grep", "-a", "-q", "litert_lm_debugger", library]).return_code != 0:
        fail("litert_lm: the library was built without the debugger")
    if elf:
        exported = _run(ctx, ["nm", "-D", "--defined-only", library], ".", "nm -D")
        prefix = "litert_lm_"
    else:
        exported = _run(ctx, ["nm", "-gU", library], ".", "nm -g")
        prefix = "_litert_lm_"
    for line in exported.splitlines():
        if not line.split(" ")[-1].startswith(prefix):
            fail("litert_lm: the library exports more than the C API: " + line)
    if elf:
        return
    loads = _run(ctx, ["otool", "-L", library], ".", "otool -L")
    for name in platform.linked:
        if "@rpath/{}.dylib ".format(name) not in loads:
            fail("litert_lm: the library does not link {}:\n{}".format(name, loads))
    symbols = _run(ctx, ["nm", "-m", library], ".", "nm -m")
    for line in symbols.splitlines():
        if "dynamically looked up" in line:
            fail("litert_lm: the library looks a symbol up at run time: " + line)

# The targets of a repository; one that cannot be built holds them all, failing.
_TARGETS = ["c_api", "version", "plugins", "litert_lm", "gpu_libraries", "gpu_library_imports"]

def _write_unavailable(ctx, why):
    """Makes the repository's targets fail with `why` when they are built.

    The Runner's selects name every platform's repository, and a build loads
    each of them: a platform this machine cannot build for must not fail the
    builds of the others.
    """
    ctx.file("unavailable.bzl", """def _impl(ctx):
    fail("litert_lm: {why}")

unavailable = rule(implementation = _impl)
""".replace("{why}", why))
    ctx.file("BUILD.bazel", 'load(":unavailable.bzl", "unavailable")\n\n' + "".join([
        'unavailable(\n    name = "{}",\n    visibility = ["//visibility:public"],\n)\n\n'.format(t)
        for t in _TARGETS
    ]))

def _plugin_path(plugin, platform, extension):
    """Returns where `plugin` is, from the directory of the runtime library."""
    if platform.frameworks:
        return "../{}.framework/{}".format(plugin, plugin)
    return "{}.{}".format(plugin, extension)

def _impl(ctx):
    platform = _PLATFORMS[ctx.attr.platform]
    if platform.elf and not ctx.os.environ.get("ANDROID_NDK_HOME"):
        _write_unavailable(ctx, "ANDROID_NDK_HOME must name an Android NDK (r28b) for " + ctx.attr.platform)
        return
    bazel = ctx.which("bazelisk")
    if bazel == None:
        fail("litert_lm: bazelisk is not installed (brew install bazelisk)")
    if ctx.which("git-lfs") == None:
        fail("litert_lm: git-lfs is not installed (brew install git-lfs)")
    commit = ctx.attr.commit
    extension = "so" if platform.elf else "dylib"
    source = _fetch_source(ctx, ctx.attr.platform)
    prebuilt = "prebuilt/{}".format(ctx.attr.platform)

    library = "liblitert-lm." + extension
    built = _build_library(ctx, bazel, source, platform)
    _run(ctx, ["cp", built, library], ".", "cp")
    _run(ctx, ["chmod", "u+w", library], ".", "chmod")
    if not platform.elf:
        # Bazel gives the library the install name bazel-out/.../liblitert-lm.dylib,
        # which a linked binary would record and never find; its id becomes
        # @rpath/liblitert-lm.dylib here, and it is signed again, since arm64
        # loads no unsigned library.
        _run(ctx, ["install_name_tool", "-id", "@rpath/" + library, library], ".", "install_name_tool")
        _run(ctx, ["codesign", "--force", "--sign", "-", library], ".", "codesign")
    _check_library(ctx, library, platform)

    # The libraries the runtime links (by @rpath, from its own directory) and
    # the GPU plugins it loads, as LiteRT-LM ships them, beside it: the
    # directory the runtime finds itself in.
    for name in platform.linked + platform.plugins:
        file = "{}.{}".format(name, extension)
        _run(ctx, ["cp", "{}/{}/{}".format(source, prebuilt, file), file], ".", "cp")
        _run(ctx, ["chmod", "u+w", file], ".", "chmod")
    if platform.frameworks:
        _wrap_in_framework(ctx, "liblitert-lm", platform.linked)
        for name in platform.linked:
            _wrap_in_framework(ctx, name, [])

        # LiteRT and LiteRT-LM open a GPU plugin by its file name alone,
        # libX.dylib, which finds the loaded image of that install name: a
        # plugin keeps its own.
        for name in platform.plugins:
            _wrap_in_framework(ctx, name, [], keep_id = True)

    for header in _HEADERS:
        ctx.symlink(source.get_child("c").get_child(header), "c/" + header)

    # The commit, for the runtime to report as its version: the C API does not
    # say, and whoever built the library knows.
    ctx.file("version.h", """// The LiteRT-LM commit this library was built from. Generated.
#ifndef LITERT_LM_VERSION_H_
#define LITERT_LM_VERSION_H_

namespace litert_lm {
inline constexpr char kCommit[] = "{commit}";
}  // namespace litert_lm

#endif  // LITERT_LM_VERSION_H_
""".replace("{commit}", commit))

    # The plugins, for the runtime to load in this order from its directory:
    # beside it, or, as frameworks, beside its own framework.
    ctx.file("plugins.h", """// The GPU plugins of this platform's LiteRT-LM, in load order. Generated.
#ifndef LITERT_LM_PLUGINS_H_
#define LITERT_LM_PLUGINS_H_

namespace litert_lm {
inline constexpr const char* kGpuPlugins[] = {%s};
}  // namespace litert_lm

#endif  // LITERT_LM_PLUGINS_H_
""" % ", ".join(['"{}"'.format(_plugin_path(p, platform, extension)) for p in platform.plugins]))
    build_file = ctx.attr._build_file_ios if platform.frameworks else ctx.attr._build_file
    ctx.template("BUILD.bazel", build_file, {
        "{commit}": commit,
        "{platform}": ctx.attr.platform,
        "{headers}": ", ".join(['"c/{}"'.format(h) for h in _HEADERS]),
        "{library}": library,
        "{linked}": ", ".join(['":{}"'.format(n) for n in platform.linked]),
        "{imports}": "\n".join([
            'cc_import(\n    name = "{}",\n    shared_library = "{}.{}",\n)\n'.format(n, n, extension)
            for n in platform.linked
        ]),
        "{plugins}": ", ".join(['"{}.{}"'.format(p, extension) for p in platform.plugins]),
        "{plugin_labels}": ", ".join(['":{}"'.format(p) for p in platform.plugins]),
        "{plugin_framework_imports}": "\n".join([
            'apple_dynamic_framework_import(\n    name = "{}",\n    bundle_only = True,\n    framework_imports = glob(["{}.framework/**"]),\n)\n'.format(p, p)
            for p in platform.plugins
        ]),
        "{plugin_imports}": "\n".join([
            'cc_import(\n    name = "{}",\n    shared_library = "{}.{}",\n)\n'.format(p, p, extension)
            for p in platform.plugins
        ]),
    })

litert_lm_library = repository_rule(
    implementation = _impl,
    attrs = {
        "commit": attr.string(
            mandatory = True,
            doc = "The LiteRT-LM commit to build, whole.",
        ),
        "url": attr.string(
            default = "https://github.com/google-ai-edge/LiteRT-LM.git",
            doc = "Where the commit is fetched from.",
        ),
        "platform": attr.string(
            mandatory = True,
            values = _PLATFORMS.keys(),
            doc = "What the library is built for, as LiteRT-LM names it.",
        ),
        "_build_file": attr.label(
            default = "//model_debugger/runner/llm/litert_lm:litert_lm.BUILD",
        ),
        "_build_file_ios": attr.label(
            default = "//model_debugger/runner/llm/litert_lm:litert_lm_ios.BUILD",
        ),
    },
    environ = ["ANDROID_NDK_HOME"],
    doc = "LiteRT-LM's C API library, built from a commit, and its GPU plugins.",
)
