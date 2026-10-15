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

#include <stdlib.h>

#include "gtest/gtest.h"

namespace model_debugger::runner::service {
namespace {

#if defined(__ANDROID__)
TEST(CacheDirectoryTest, IsTheTemporaryDirectoryOnAndroid) {
  setenv("TMPDIR", "/data/local/tmp/x", /*overwrite=*/1);
  EXPECT_EQ(CacheDirectory(), "/data/local/tmp/x/model_debugger");
}

TEST(CacheDirectoryTest, IsAdbsWithoutATemporaryDirectory) {
  unsetenv("TMPDIR");
  EXPECT_EQ(CacheDirectory(), "/data/local/tmp/model_debugger");
}
#elif defined(__APPLE__)
TEST(CacheDirectoryTest, IsTheUsersCachesOnApplePlatforms) {
  setenv("HOME", "/Users/someone", /*overwrite=*/1);
  EXPECT_EQ(CacheDirectory(), "/Users/someone/Library/Caches/model_debugger");
}

TEST(CacheDirectoryTest, IsNoneWithoutAHome) {
  unsetenv("HOME");
  EXPECT_TRUE(CacheDirectory().empty());
}
#elif defined(__linux__)
TEST(CacheDirectoryTest, IsTheUsersCachesOnLinux) {
  unsetenv("XDG_CACHE_HOME");
  setenv("HOME", "/home/someone", /*overwrite=*/1);
  EXPECT_EQ(CacheDirectory(), "/home/someone/.cache/model_debugger");
}

TEST(CacheDirectoryTest, UsesXdgCacheHomeWhenSetOnLinux) {
  setenv("XDG_CACHE_HOME", "/tmp/xdg_cache", /*overwrite=*/1);
  EXPECT_EQ(CacheDirectory(), "/tmp/xdg_cache/model_debugger");
  unsetenv("XDG_CACHE_HOME");
}

TEST(CacheDirectoryTest, IsNoneWithoutAHome) {
  unsetenv("XDG_CACHE_HOME");
  unsetenv("HOME");
  EXPECT_TRUE(CacheDirectory().empty());
}
#endif

}  // namespace
}  // namespace model_debugger::runner::service
