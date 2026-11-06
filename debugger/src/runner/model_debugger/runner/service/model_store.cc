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

#include "model_debugger/runner/service/model_store.h"

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <string>
#include <string_view>
#include <system_error>  // NOLINT(build/c++11)
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace model_debugger::runner::service {
namespace {

// The extension a model file is stored with: what the one runtime, LiteRT-LM,
// knows the format by.
constexpr std::string_view kModelExtension = ".litertlm";

// What the system says of errno value `error`.
std::string Reason(int error) { return std::generic_category().message(error); }

// Closes `fd`, which lets go of its lock; does nothing for -1.
void Close(int fd) {
  if (fd != -1) close(fd);
}

// The store directory locked, for as long as it lives: each step that finds,
// holds, adds or deletes a file takes it. Every step under it is short, so
// another process waits for it without a deadline.
class DirectoryLock {
 public:
  // Waits for the lock on `dir`. Internal if it cannot be taken.
  static absl::StatusOr<DirectoryLock> Take(const std::filesystem::path& dir) {
    const int fd = open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd == -1) {
      return absl::InternalError(
          absl::StrCat("Cannot open ", dir.string(), ": ", Reason(errno)));
    }
    int locked;
    do {
      locked = flock(fd, LOCK_EX);
    } while (locked == -1 && errno == EINTR);
    if (locked == -1) {
      const int error = errno;
      close(fd);
      return absl::InternalError(
          absl::StrCat("Cannot lock ", dir.string(), ": ", Reason(error)));
    }
    return DirectoryLock(fd);
  }

  DirectoryLock(DirectoryLock&& other) noexcept
      : fd_(std::exchange(other.fd_, -1)) {}
  DirectoryLock& operator=(DirectoryLock&&) = delete;
  ~DirectoryLock() { Close(fd_); }

 private:
  explicit DirectoryLock(int fd) : fd_(fd) {}

  int fd_;
};

// Opens `path` and locks it shared, as a hold; called with the directory
// locked, so no KeepOnly is between finding the file and holding it. NotFound
// if there is no such file.
absl::StatusOr<int> OpenHeld(const std::filesystem::path& path) {
  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd == -1) {
    const int error = errno;
    if (error == ENOENT) {
      return absl::NotFoundError(
          absl::StrCat(path.string(), " is not in the model store."));
    }
    return absl::InternalError(
        absl::StrCat("Cannot open ", path.string(), ": ", Reason(error)));
  }
  // Only a KeepOnly locks a model exclusively, and only under the directory's
  // lock, which the caller holds: this never waits.
  if (flock(fd, LOCK_SH | LOCK_NB) == -1) {
    const int error = errno;
    close(fd);
    return absl::InternalError(
        absl::StrCat("Cannot hold ", path.string(), ": ", Reason(error)));
  }
  return fd;
}

}  // namespace

ModelStore::Hold::Hold(Hold&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)) {}

ModelStore::Hold& ModelStore::Hold::operator=(Hold&& other) noexcept {
  if (this != &other) {
    Close(fd_);
    fd_ = std::exchange(other.fd_, -1);
  }
  return *this;
}

ModelStore::Hold::~Hold() { Close(fd_); }

ModelStore::Upload::Upload(Upload&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)),
      path_(std::move(other.path_)),
      sha256_(std::move(other.sha256_)) {}

ModelStore::Upload& ModelStore::Upload::operator=(Upload&& other) noexcept {
  if (this != &other) {
    Discard();
    fd_ = std::exchange(other.fd_, -1);
    path_ = std::move(other.path_);
    sha256_ = std::move(other.sha256_);
  }
  return *this;
}

ModelStore::Upload::~Upload() { Discard(); }

void ModelStore::Upload::Discard() {
  if (fd_ == -1) return;
  // Deleted before it is let go of: no KeepOnly finds it unheld meanwhile,
  // and no other upload takes the name while the file exists.
  if (unlink(path_.c_str()) == -1 && errno != ENOENT) {
    LOG(WARNING) << "Cannot delete the partial upload " << path_.string()
                 << ": " << Reason(errno);
  }
  close(std::exchange(fd_, -1));
}

absl::Status ModelStore::Upload::Append(std::string_view bytes) {
  while (!bytes.empty()) {
    const ssize_t written = write(fd_, bytes.data(), bytes.size());
    if (written == -1) {
      const int error = errno;
      if (error == EINTR) continue;
      const std::string message =
          absl::StrCat("Cannot write the model: ", Reason(error));
      return error == ENOSPC ? absl::ResourceExhaustedError(message)
                             : absl::InternalError(message);
    }
    bytes.remove_prefix(static_cast<size_t>(written));
  }
  return absl::OkStatus();
}

std::filesystem::path ModelStore::ModelPath(std::string_view sha256) const {
  return dir_ / absl::StrCat(sha256, kModelExtension);
}

void ModelStore::KeepOnly(std::string_view sha256) {
  absl::StatusOr<DirectoryLock> lock = DirectoryLock::Take(dir_);
  if (!lock.ok()) {
    LOG(WARNING) << "The model store keeps every model: "
                 << lock.status().message();
    return;
  }
  const std::filesystem::path keep = ModelPath(sha256);
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator(dir_, error)) {
    const std::filesystem::path& path = entry.path();
    if (path == keep) continue;
    // A model or a partial upload is a file, or a link to one; the store has
    // nothing else of its own.
    std::error_code type_error;
    const std::filesystem::file_type type =
        entry.symlink_status(type_error).type();
    if (type != std::filesystem::file_type::regular &&
        type != std::filesystem::file_type::symlink) {
      continue;
    }
    // A file this Runner cannot open, no Runner sharing the store holds
    // either: it goes too.
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd != -1 && flock(fd, LOCK_EX | LOCK_NB) == -1) {
      const int lock_error = errno;
      close(fd);
      if (lock_error == EWOULDBLOCK) continue;  // Held: in use.
      LOG(WARNING) << "Cannot tell whether " << path.string()
                   << " is in use, so it stays: " << Reason(lock_error);
      continue;
    }
    if (unlink(path.c_str()) == -1) {
      LOG(WARNING) << "Cannot delete " << path.string()
                   << " from the model store: " << Reason(errno);
    }
    Close(fd);
  }
  if (error) {
    LOG(WARNING) << "Cannot list the model store " << dir_.string() << ": "
                 << error.message();
  }
}

absl::StatusOr<ModelStore::Hold> ModelStore::HoldModel(
    std::string_view sha256) {
  absl::StatusOr<DirectoryLock> lock = DirectoryLock::Take(dir_);
  if (!lock.ok()) return lock.status();
  absl::StatusOr<int> fd = OpenHeld(ModelPath(sha256));
  if (!fd.ok()) return fd.status();
  return Hold(*fd);
}

absl::StatusOr<ModelStore::Upload> ModelStore::StartUpload(
    std::string_view sha256) {
  // A name of its own, so that two uploads of one model never write one file.
  std::string name =
      (dir_ / absl::StrCat(sha256, kModelExtension, ".part.XXXXXX")).string();
  absl::StatusOr<DirectoryLock> lock = DirectoryLock::Take(dir_);
  if (!lock.ok()) return lock.status();
  const int fd = mkostemp(name.data(), O_CLOEXEC);
  if (fd == -1) {
    return absl::InternalError(
        absl::StrCat("Cannot create ", name, ": ", Reason(errno)));
  }
  // Locked before the directory is let go of: no KeepOnly finds it unheld.
  if (flock(fd, LOCK_EX | LOCK_NB) == -1) {
    const int error = errno;
    unlink(name.c_str());
    close(fd);
    return absl::InternalError(
        absl::StrCat("Cannot lock ", name, ": ", Reason(error)));
  }
  return Upload(fd, std::move(name), std::string(sha256));
}

absl::StatusOr<ModelStore::Hold> ModelStore::Keep(Upload upload) {
  absl::StatusOr<DirectoryLock> lock = DirectoryLock::Take(dir_);
  if (!lock.ok()) return lock.status();
  const std::filesystem::path model = ModelPath(upload.sha256_);
  std::error_code error;
  if (std::filesystem::exists(model, error)) {
    // Another upload kept the model first; a Runner may hold it already.
    upload.Discard();
    absl::StatusOr<int> fd = OpenHeld(model);
    if (!fd.ok()) return fd.status();
    return Hold(*fd);
  }
  if (rename(upload.path_.c_str(), model.c_str()) == -1) {
    return absl::InternalError(
        absl::StrCat("Cannot keep the model: ", Reason(errno)));
  }
  // The upload's exclusive lock goes before the hold's shared one is taken,
  // still under the directory's lock: no KeepOnly is between them.
  close(std::exchange(upload.fd_, -1));
  absl::StatusOr<int> fd = OpenHeld(model);
  if (!fd.ok()) return fd.status();
  return Hold(*fd);
}

}  // namespace model_debugger::runner::service
