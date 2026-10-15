// Copyright 2026 The AI Edge Model Explorer Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Where a Runner keeps everything on this machine: model files and the
// directories of its runs. Nobody passes it; each platform has its own.

#ifndef MODEL_DEBUGGER_RUNNER_SERVICE_CACHE_DIRECTORY_H_
#define MODEL_DEBUGGER_RUNNER_SERVICE_CACHE_DIRECTORY_H_

#include <filesystem>

namespace model_debugger::runner::service {

// Returns model_debugger/ under the user's cache directory on this platform:
//   macOS, iOS  $HOME/Library/Caches (on iOS, the App's own, in its sandbox)
//   Android     $TMPDIR, which adb's shell sets and the App sets to its
//               Context.getCacheDir(), else /data/local/tmp
// Returns an empty path when the environment names none: a RunnerService
// cannot then be made.
std::filesystem::path CacheDirectory();

}  // namespace model_debugger::runner::service

#endif  // MODEL_DEBUGGER_RUNNER_SERVICE_CACHE_DIRECTORY_H_
