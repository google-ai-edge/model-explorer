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

// What the tests read from their environment, and where they put files. The
// runtime's, the service's and the Runner's tests need the same.

#ifndef MODEL_DEBUGGER_RUNNER_TESTING_TEST_UTILS_H_
#define MODEL_DEBUGGER_RUNNER_TESTING_TEST_UTILS_H_

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>

#include "gtest/gtest.h"
#include "model_debugger/runner/service/cache_directory.h"
#include "model_debugger/v1/runner.pb.h"

namespace model_debugger::runner {

// A fresh directory for this run, under googletest's temporary directory:
// Bazel's TEST_TMPDIR, which is this run's alone, so that runs side by side do
// not share files.
inline std::filesystem::path TempDir(std::string_view name) {
  const std::filesystem::path dir =
      std::filesystem::path(::testing::TempDir()) / name;
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir;
}

// Points every variable a Runner finds its cache directory by at a directory
// of this run's own, and returns the Runner's cache directory under it.
inline std::filesystem::path UseTestCacheDirectory() {
  const std::filesystem::path home = TempDir("home");
  unsetenv("XDG_CACHE_HOME");
  setenv("HOME", home.c_str(), /*overwrite=*/1);    // macOS, iOS, Linux
  setenv("TMPDIR", home.c_str(), /*overwrite=*/1);  // Android
  return service::CacheDirectory();
}

// A Runner keeps everything under HOME: never the real one in a test.
class TestHome : public testing::Environment {
 public:
  void SetUp() override { UseTestCacheDirectory(); }
};

// The platform this test runs on, as HostInfo and the C facade name it.
#if defined(__ANDROID__)
inline constexpr v1::HostInfo::Platform kPlatform = v1::HostInfo::ANDROID;
inline constexpr char kPlatformName[] = "Android";
#elif defined(__APPLE__) && TARGET_OS_IPHONE
inline constexpr v1::HostInfo::Platform kPlatform = v1::HostInfo::IOS;
inline constexpr char kPlatformName[] = "iOS";
#elif defined(__APPLE__)
inline constexpr v1::HostInfo::Platform kPlatform = v1::HostInfo::MACOS;
inline constexpr char kPlatformName[] = "macOS";
#elif defined(__linux__)
inline constexpr v1::HostInfo::Platform kPlatform = v1::HostInfo::LINUX;
inline constexpr char kPlatformName[] = "Linux";
#else
#error "The Runner is built for Android, iOS, Linux and macOS."
#endif

// CPU, or GPU when RUNNER_BACKEND is "gpu", which the GPU variant of a test
// (testing/model_test.bzl) sets.
inline v1::Device DeviceFromEnvironment() {
  const char* backend = std::getenv("RUNNER_BACKEND");
  return backend != nullptr && std::string(backend) == "gpu" ? v1::GPU
                                                             : v1::CPU;
}

}  // namespace model_debugger::runner

#endif  // MODEL_DEBUGGER_RUNNER_TESTING_TEST_UTILS_H_
