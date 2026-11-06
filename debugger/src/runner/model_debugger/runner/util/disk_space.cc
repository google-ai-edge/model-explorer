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

#include <sys/stat.h>
#include <sys/statvfs.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace model_debugger::runner {
namespace {

// Returns the directory part of `path`: "" if it has none, "/" for the root.
std::string ParentPath(absl::string_view path) {
  const size_t slash = path.find_last_of('/');
  if (slash == absl::string_view::npos) return "";
  if (slash == 0) return "/";
  return std::string(path.substr(0, slash));
}

bool IsDirectory(const std::string& path) {
  struct stat info;
  return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

}  // namespace

std::string OutOfRoomMessage(absl::string_view file, std::uintmax_t needed) {
  std::string dir = ParentPath(file);
  while (!dir.empty() && dir != "/" && !IsDirectory(dir)) {
    dir = ParentPath(dir);
  }
  struct statvfs disk;
  if (statvfs(dir.empty() ? "." : dir.c_str(), &disk) != 0) return "";
  const std::uintmax_t available =
      static_cast<std::uintmax_t>(disk.f_bavail) * disk.f_frsize;
  if (available >= needed) return "";
  return absl::StrCat("runner: no space left on device while writing ", file,
                      ": ", needed, " bytes did not fit; the disk has ",
                      available, " bytes free.\n");
}

void ExitIfOutOfRoom(absl::string_view file, std::uintmax_t needed) {
  const std::string line = OutOfRoomMessage(file, needed);
  if (line.empty()) return;
  std::fputs(line.c_str(), stderr);
  std::fflush(stderr);
  std::_Exit(EXIT_FAILURE);
}

}  // namespace model_debugger::runner
