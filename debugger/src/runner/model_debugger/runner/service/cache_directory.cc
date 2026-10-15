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

#include "model_debugger/runner/service/cache_directory.h"

#include <cstdlib>
#include <filesystem>

namespace model_debugger::runner::service {
namespace {

// The variable's value, or empty if it is unset or empty.
std::filesystem::path FromEnvironment(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return {};
  return value;
}

// The platform's cache directory, before model_debugger/.
std::filesystem::path PlatformCacheDirectory() {
#if defined(__APPLE__)
  const std::filesystem::path home = FromEnvironment("HOME");
  return home.empty() ? home : home / "Library" / "Caches";
#elif defined(__ANDROID__)
  // adb's shell sets TMPDIR; a command run through adb may not.
  const std::filesystem::path tmp = FromEnvironment("TMPDIR");
  return tmp.empty() ? std::filesystem::path("/data/local/tmp") : tmp;
#elif defined(__linux__)
  if (const std::filesystem::path xdg = FromEnvironment("XDG_CACHE_HOME");
      !xdg.empty()) {
    return xdg;
  }
  const std::filesystem::path home = FromEnvironment("HOME");
  return home.empty() ? home : home / ".cache";
#else
#error "cache_directory.cc knows no cache directory for this platform."
#endif
}

}  // namespace

std::filesystem::path CacheDirectory() {
  const std::filesystem::path root = PlatformCacheDirectory();
  return root.empty() ? root : root / "model_debugger";
}

}  // namespace model_debugger::runner::service
