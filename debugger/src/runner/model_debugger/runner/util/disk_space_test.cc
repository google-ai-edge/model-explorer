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

#include "model_debugger/runner/util/disk_space.h"

#include <sys/statvfs.h>
#include <unistd.h>

#include <cstdint>
#include <limits>
#include <string>

#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace model_debugger::runner {
namespace {

std::string TempFile(absl::string_view relative_path) {
  return absl::StrCat(::testing::TempDir(), "/", relative_path);
}

class ScopedChdir {
 public:
  explicit ScopedChdir(const char* target) {
    char buf[4096];
    if (getcwd(buf, sizeof(buf)) != nullptr) {
      previous_ = buf;
    }
    moved_ = (chdir(target) == 0);
  }

  ScopedChdir(const ScopedChdir&) = delete;
  ScopedChdir& operator=(const ScopedChdir&) = delete;
  ScopedChdir(ScopedChdir&&) = delete;
  ScopedChdir& operator=(ScopedChdir&&) = delete;

  ~ScopedChdir() {
    if (!previous_.empty()) {
      (void)chdir(previous_.c_str());
    }
  }

  bool moved() const { return moved_; }

 private:
  std::string previous_;
  bool moved_ = false;
};

TEST(DiskSpaceTest, AWriteTheDiskHasRoomForReturns) {
  ExitIfOutOfRoom(TempFile("a.safetensors"), 1);
  ExitIfOutOfRoom(TempFile("turn-1/b/a.safetensors"), 1);
  ExitIfOutOfRoom("a.safetensors", 1);

  struct statvfs temp_disk;
  ASSERT_EQ(statvfs(::testing::TempDir().c_str(), &temp_disk), 0);
  if (temp_disk.f_bavail >= 4 && temp_disk.f_frsize > 2) {
    const std::uintmax_t half_available =
        (static_cast<std::uintmax_t>(temp_disk.f_bavail) / 2) *
        temp_disk.f_frsize;
    EXPECT_EQ(
        OutOfRoomMessage(TempFile("turn-1/b/a.safetensors"), half_available),
        "");
  }
}

TEST(DiskSpaceTest, OutOfRoomMessageFormatsMissingSpaceAndWalksParents) {
  const std::string file = TempFile("turn-1/b/a.safetensors");
  const std::string message =
      OutOfRoomMessage(file, std::numeric_limits<std::uintmax_t>::max());
  EXPECT_NE(message.find("runner: no space left on device while writing "),
            std::string::npos);
  EXPECT_NE(message.find("/turn-1/b/a.safetensors: 18446744073709551615 bytes "
                         "did not fit; the disk has "),
            std::string::npos);

#ifdef __linux__
  struct statvfs proc_disk;
  struct statvfs root_disk;
  if (statvfs("/proc", &proc_disk) == 0 && proc_disk.f_bavail == 0 &&
      statvfs("/", &root_disk) == 0 && root_disk.f_bavail > 0) {
    // "/proc" has 0 free blocks while "." (TempDir) and "/" have room:
    // distinguishes "/proc/missing/..." from "." and "/" from "".
    EXPECT_EQ(OutOfRoomMessage("/proc/missing/a.safetensors", 0), "");
    EXPECT_EQ(
        OutOfRoomMessage("/proc/missing/a.safetensors", 1),
        "runner: no space left on device while writing "
        "/proc/missing/a.safetensors: 1 bytes did not fit; the disk has 0 "
        "bytes free.\n");

    ScopedChdir in_proc("/proc");
    if (in_proc.moved()) {
      EXPECT_EQ(OutOfRoomMessage(
                    "/nonexistent_model_debugger_test_dir/a.safetensors", 1),
                "");
      EXPECT_NE(OutOfRoomMessage("a.safetensors", 1), "");
    }
  }
#endif
}

TEST(DiskSpaceDeathTest, AWriteTheDiskHasNoRoomForEndsTheRunnerSayingWhy) {
  // More than any disk holds; the file's directory is not there yet.
  const std::string file = TempFile("turn-1/a.safetensors");
  EXPECT_EXIT(
      ExitIfOutOfRoom(file, std::numeric_limits<std::uintmax_t>::max()),
      ::testing::ExitedWithCode(1),
      "runner: no space left on device while writing .*/turn-1/"
      "a\\.safetensors: 18446744073709551615 bytes did not fit; the disk has "
      "[0-9]+ bytes free\\.");
}

}  // namespace
}  // namespace model_debugger::runner
