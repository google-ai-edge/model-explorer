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

#ifndef MODEL_DEBUGGER_RUNNER_UTIL_DISK_SPACE_H_
#define MODEL_DEBUGGER_RUNNER_UTIL_DISK_SPACE_H_

#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"

namespace model_debugger::runner {

// Returns the error line written to stderr when the disk `file` is to be
// written to has fewer than `needed` bytes free, or "" when there is room or
// the free space cannot be determined.
std::string OutOfRoomMessage(absl::string_view file, std::uintmax_t needed);

// Ends the process, saying why on stderr, if the disk `file` is to be written
// to has fewer than `needed` bytes free. The disk is that of the nearest
// existing directory above `file`. Returns if the free space is unknown.
void ExitIfOutOfRoom(absl::string_view file, std::uintmax_t needed);

}  // namespace model_debugger::runner

#endif  // MODEL_DEBUGGER_RUNNER_UTIL_DISK_SPACE_H_
