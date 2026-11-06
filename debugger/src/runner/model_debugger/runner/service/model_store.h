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

// The model files a Runner keeps on its machine, and the rule that keeps only
// the one in use.

#ifndef MODEL_DEBUGGER_RUNNER_SERVICE_MODEL_STORE_H_
#define MODEL_DEBUGGER_RUNNER_SERVICE_MODEL_STORE_H_

#include <filesystem>  // NOLINT(build/c++17)
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace model_debugger::runner::service {

// A directory of model files, each named `<sha256>.litertlm`. Every Runner
// that finds the same cache directory shares it: on macOS, the command-line
// Runners and the App of one user.
//
// The store keeps one model: KeepOnly deletes every file but the model named
// and those a Runner holds, a model it loaded or is about to load (Hold) or
// one it is uploading (Upload). A hold is an advisory lock, flock(2), on the
// file's open description, so the system drops it when the holder exits,
// crashed or not, and the next KeepOnly deletes what that process left. A
// KeepOnly in the same process skips its holds too. Each step that finds,
// holds, adds or deletes a file locks the directory itself, so a file is never
// deleted between being found and being held.
//
// Thread-safe. Any number of instances, in any processes, may share a
// directory, which must exist.
class ModelStore {
 public:
  // A model file held open: while it lives, no KeepOnly deletes it.
  class Hold {
   public:
    Hold(Hold&& other) noexcept;
    Hold& operator=(Hold&& other) noexcept;
    ~Hold();

   private:
    friend class ModelStore;
    explicit Hold(int fd) : fd_(fd) {}

    int fd_;  // -1 once moved from.
  };

  // A model being written under a name of its own, which no KeepOnly
  // deletes. Unless kept, it is deleted when destroyed.
  class Upload {
   public:
    Upload(Upload&& other) noexcept;
    Upload& operator=(Upload&& other) noexcept;
    ~Upload();

    // Appends `bytes` to the file. ResourceExhausted if the disk is full,
    // Internal if writing fails otherwise; either names the system's reason.
    absl::Status Append(std::string_view bytes);

   private:
    friend class ModelStore;
    Upload(int fd, std::filesystem::path path, std::string sha256)
        : fd_(fd), path_(std::move(path)), sha256_(std::move(sha256)) {}

    // Deletes the file and lets go of it; does nothing once moved from.
    void Discard();

    int fd_;  // -1 once moved from or kept.
    std::filesystem::path path_;
    std::string sha256_;  // The model it will be kept as.
  };

  explicit ModelStore(std::filesystem::path dir) : dir_(std::move(dir)) {}

  // Where the model of `sha256` is kept, if it is.
  std::filesystem::path ModelPath(std::string_view sha256) const;

  // Deletes every file in the store but the model of `sha256` and the files a
  // Runner holds: other models, and partial uploads whose Runner is gone. A
  // file that cannot be deleted is logged with the system's reason and left;
  // nothing fails.
  void KeepOnly(std::string_view sha256);

  // Holds the model of `sha256`. NotFound if the store has none; Internal if
  // the file cannot be opened or locked.
  absl::StatusOr<Hold> HoldModel(std::string_view sha256);

  // Starts writing the model of `sha256`. Internal if the file cannot be
  // made.
  absl::StatusOr<Upload> StartUpload(std::string_view sha256);

  // Keeps what `upload` wrote as its model and holds it. A model of that
  // sha256 that came meanwhile, from another upload, is kept instead, and
  // this one deleted: a held file is never replaced. Internal if the file
  // cannot be renamed or held.
  absl::StatusOr<Hold> Keep(Upload upload);

 private:
  const std::filesystem::path dir_;
};

}  // namespace model_debugger::runner::service

#endif  // MODEL_DEBUGGER_RUNNER_SERVICE_MODEL_STORE_H_
