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

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <functional>
#include <iterator>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>

#include "model_debugger/runner/testing/test_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/log_severity.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace model_debugger::runner::service {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

// The model a test names, and two others in the store.
constexpr std::string_view kSha256 =
    "9f3a7c1e5b2d4a6f8e0c1b3d5f7a9c2e4b6d8f0a1c3e5b7d9f2a4c6e8b0d1f3a";
constexpr std::string_view kOther =
    "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
constexpr std::string_view kThird =
    "0000000000000000000000000000000000000000000000000000000000000000";

std::string ModelName(std::string_view sha256) {
  return std::string(sha256) + ".litertlm";
}

void Write(const std::filesystem::path& file, std::string_view bytes) {
  std::ofstream(file, std::ios::binary) << bytes;
}

std::string Read(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), {});
}

// Another process, standing for another Runner sharing the store: it runs
// `take`, which takes holds and must leave them held, then waits to be
// killed, which lets go of them as a crash would.
class OtherProcess {
 public:
  // Starts the process; returns once `take` returned in it. ok() is false if
  // `take` failed or the process could not start.
  explicit OtherProcess(const std::function<bool()>& take) {
    int pipe_fds[2];
    if (pipe(pipe_fds) != 0) return;
    pid_ = fork();
    if (pid_ == 0) {
      close(pipe_fds[0]);
      const char taken = take() ? 'y' : 'n';
      (void)write(pipe_fds[1], &taken, 1);
      for (;;) pause();
    }
    close(pipe_fds[1]);
    char taken = 'n';
    ok_ = pid_ > 0 && read(pipe_fds[0], &taken, 1) == 1 && taken == 'y';
    close(pipe_fds[0]);
  }

  ~OtherProcess() { Kill(); }

  bool ok() const { return ok_; }

  // Kills the process and waits until it is gone, its holds with it.
  void Kill() {
    if (pid_ <= 0) return;
    kill(pid_, SIGKILL);
    waitpid(pid_, nullptr, 0);
    pid_ = -1;
  }

 private:
  pid_t pid_ = -1;
  bool ok_ = false;
};

class ModelStoreTest : public testing::Test {
 protected:
  ModelStoreTest() : dir_(TempDir("models")), store_(dir_) {}

  // The names of the files in the store.
  std::set<std::string> Names() const {
    std::set<std::string> names;
    for (const auto& entry : std::filesystem::directory_iterator(dir_)) {
      names.insert(entry.path().filename());
    }
    return names;
  }

  // An upload of `sha256` with `bytes` written, ready to keep.
  ModelStore::Upload Uploaded(std::string_view sha256, std::string_view bytes) {
    absl::StatusOr<ModelStore::Upload> upload = store_.StartUpload(sha256);
    EXPECT_TRUE(upload.ok()) << upload.status();
    EXPECT_TRUE(upload->Append(bytes).ok());
    return *std::move(upload);
  }

  const std::filesystem::path dir_;
  ModelStore store_;
};

TEST_F(ModelStoreTest, KeepsTheModelNamedAndDeletesEveryOther) {
  Write(dir_ / ModelName(kSha256), "named");
  Write(dir_ / ModelName(kOther), "other");
  Write(dir_ / ModelName(kThird), "third");
  store_.KeepOnly(kSha256);
  EXPECT_THAT(Names(), ElementsAre(ModelName(kSha256)));
  EXPECT_EQ(Read(dir_ / ModelName(kSha256)), "named") << "used as it is";
}

TEST_F(ModelStoreTest, KeepsNothingWhenTheModelNamedIsNotThere) {
  Write(dir_ / ModelName(kOther), "other");
  store_.KeepOnly(kSha256);
  EXPECT_THAT(Names(), IsEmpty());
}

TEST_F(ModelStoreTest, KeepsAModelHeldInThisProcessUntilItIsLetGo) {
  Write(dir_ / ModelName(kOther), "other");
  {
    absl::StatusOr<ModelStore::Hold> hold = store_.HoldModel(kOther);
    ASSERT_TRUE(hold.ok()) << hold.status();
    ModelStore(dir_).KeepOnly(kSha256);
    EXPECT_THAT(Names(), ElementsAre(ModelName(kOther)))
        << "held by this process, through another instance";
    std::optional<ModelStore::Hold> moved(*std::move(hold));
    store_.KeepOnly(kSha256);
    EXPECT_THAT(Names(), ElementsAre(ModelName(kOther))) << "moved, still held";
  }
  store_.KeepOnly(kSha256);
  EXPECT_THAT(Names(), IsEmpty());
}

TEST_F(ModelStoreTest, KeepsAModelAnotherProcessHoldsUntilItIsGone) {
  Write(dir_ / ModelName(kOther), "other");
  Write(dir_ / ModelName(kThird), "third");
  OtherProcess runner([&] {
    // Never let go of: the process is killed with it held.
    return (new absl::StatusOr<ModelStore::Hold>(
                ModelStore(dir_).HoldModel(kOther)))
        ->ok();
  });
  ASSERT_TRUE(runner.ok());
  store_.KeepOnly(kSha256);
  EXPECT_THAT(Names(), ElementsAre(ModelName(kOther)));
  runner.Kill();
  store_.KeepOnly(kSha256);
  EXPECT_THAT(Names(), IsEmpty()) << "its holder crashed";
}

TEST_F(ModelStoreTest, KeepsAnUploadAnotherProcessWritesUntilItIsGone) {
  OtherProcess runner([&] {
    absl::StatusOr<ModelStore::Upload> upload =
        ModelStore(dir_).StartUpload(kOther);
    if (!upload.ok() || !upload->Append("half").ok()) return false;
    new ModelStore::Upload(*std::move(upload));  // Never kept nor deleted.
    return true;
  });
  ASSERT_TRUE(runner.ok());
  store_.KeepOnly(kSha256);
  ASSERT_EQ(Names().size(), 1u) << "the upload in progress";
  EXPECT_THAT(*Names().begin(), HasSubstr(ModelName(kOther) + ".part"));
  runner.Kill();
  store_.KeepOnly(kSha256);
  EXPECT_THAT(Names(), IsEmpty()) << "its writer crashed";
}

TEST_F(ModelStoreTest, DeletesPartialUploadsOfTheModelNamedToo) {
  Write(dir_ / ModelName(kSha256), "named");
  // Left by Runners that crashed while writing, this one's name and the
  // name Runners wrote to before.
  Write(dir_ / (ModelName(kSha256) + ".part.Ab12Cd"), "half");
  Write(dir_ / (ModelName(kOther) + ".part"), "half");
  store_.KeepOnly(kSha256);
  EXPECT_THAT(Names(), ElementsAre(ModelName(kSha256)));
}

TEST_F(ModelStoreTest, DeletesALinkToAModelNotWhatItLinksTo) {
  const std::filesystem::path elsewhere = TempDir("elsewhere") / "model";
  Write(elsewhere, "other");
  std::filesystem::create_symlink(elsewhere, dir_ / ModelName(kOther));
  std::filesystem::create_symlink(dir_ / "gone", dir_ / ModelName(kThird));
  store_.KeepOnly(kSha256);
  EXPECT_THAT(Names(), IsEmpty()) << "a dangling link goes too";
  EXPECT_EQ(Read(elsewhere), "other");
}

TEST_F(ModelStoreTest, LeavesWhatIsNotAFile) {
  std::filesystem::create_directory(dir_ / "not-a-model");
  store_.KeepOnly(kSha256);
  EXPECT_THAT(Names(), ElementsAre("not-a-model"));
}

TEST_F(ModelStoreTest, LogsAFileItCannotDeleteAndGoesOn) {
  Write(dir_ / ModelName(kSha256), "named");
  Write(dir_ / ModelName(kOther), "other");
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, _,
                       HasSubstr(ModelName(kOther) +
                                 " from the model store: Permission denied")));
  log.StartCapturingLogs();
  // A directory nothing can be deleted from.
  std::filesystem::permissions(dir_, std::filesystem::perms::owner_write,
                               std::filesystem::perm_options::remove);
  store_.KeepOnly(kSha256);
  std::filesystem::permissions(dir_, std::filesystem::perms::owner_write,
                               std::filesystem::perm_options::add);
  EXPECT_THAT(Names(), ElementsAre(ModelName(kOther), ModelName(kSha256)));
  EXPECT_TRUE(store_.HoldModel(kSha256).ok()) << "the model named is usable";
}

TEST_F(ModelStoreTest, HoldsOnlyAModelThatIsThere) {
  EXPECT_TRUE(absl::IsNotFound(store_.HoldModel(kSha256).status()));
  Write(dir_ / ModelName(kSha256), "named");
  EXPECT_TRUE(store_.HoldModel(kSha256).ok());
}

TEST_F(ModelStoreTest, AnUploadKeptIsTheModelAndHeld) {
  absl::StatusOr<ModelStore::Hold> hold =
      store_.Keep(Uploaded(kOther, "other"));
  ASSERT_TRUE(hold.ok()) << hold.status();
  EXPECT_THAT(Names(), ElementsAre(ModelName(kOther)));
  EXPECT_EQ(Read(store_.ModelPath(kOther)), "other");
  store_.KeepOnly(kSha256);
  EXPECT_THAT(Names(), ElementsAre(ModelName(kOther))) << "held";
}

TEST_F(ModelStoreTest, AnUploadNotKeptLeavesNothing) {
  {
    ModelStore::Upload upload = Uploaded(kOther, "half");
  }
  EXPECT_THAT(Names(), IsEmpty());
}

TEST_F(ModelStoreTest, TwoUploadsOfOneModelWriteFilesOfTheirOwn) {
  ModelStore::Upload first = Uploaded(kOther, "first");
  ModelStore::Upload second = Uploaded(kOther, "other");
  EXPECT_EQ(Names().size(), 2u);
  absl::StatusOr<ModelStore::Hold> kept = store_.Keep(std::move(first));
  ASSERT_TRUE(kept.ok()) << kept.status();
  absl::StatusOr<ModelStore::Hold> again = store_.Keep(std::move(second));
  ASSERT_TRUE(again.ok()) << again.status();
  EXPECT_THAT(Names(), ElementsAre(ModelName(kOther)));
  EXPECT_EQ(Read(store_.ModelPath(kOther)), "first")
      << "the model kept first, which a Runner may hold, is not replaced";
}

}  // namespace
}  // namespace model_debugger::runner::service
