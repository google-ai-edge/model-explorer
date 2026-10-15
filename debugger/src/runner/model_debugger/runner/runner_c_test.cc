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

// The C facade, as the Apps use it: a buffer in, text out, nothing to free.

#include "model_debugger/runner/runner_c.h"

#include <cstddef>
#include <string>

#include "model_debugger/runner/testing/test_utils.h"
#include "model_debugger/runner/util/json.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"

namespace model_debugger::runner {
namespace {

const testing::Environment* const kTestHome =
    testing::AddGlobalTestEnvironment(new TestHome);

TEST(RunnerCTest, StartFailureExplainsItself) {
  char error[256] = "";
  EXPECT_EQ(model_debugger_runner_start("", error, sizeof(error)), nullptr);
  EXPECT_NE(std::string(error).find("listen"), std::string::npos) << error;
}

TEST(RunnerCTest, AnErrorIsCutToTheBuffer) {
  char error[8] = "";
  EXPECT_EQ(model_debugger_runner_start("", error, sizeof(error)), nullptr);
  EXPECT_EQ(std::string(error).size(), 7u);
}

TEST(RunnerCTest, StateIsJsonOfTheRunner) {
  char error[256] = "";
  ModelDebuggerRunner* runner =
      model_debugger_runner_start("127.0.0.1:0", error, sizeof(error));
  ASSERT_NE(runner, nullptr) << error;
  char buffer[1024];
  const size_t length =
      model_debugger_runner_get_state(runner, buffer, sizeof(buffer));
  ASSERT_LT(length, sizeof(buffer));
  EXPECT_EQ(length, std::string(buffer).size());
  absl::StatusOr<Json> state = ParseJson(buffer);
  ASSERT_TRUE(state.ok()) << buffer;
  EXPECT_EQ((*state)["platform"], kPlatformName);
  EXPECT_FALSE((*state)["os_version"].get<std::string>().empty());
  EXPECT_FALSE((*state)["device"].get<std::string>().empty());
  EXPECT_EQ((*state)["address"], "127.0.0.1");
  EXPECT_NE((*state)["port"], 0);
  EXPECT_EQ((*state)["client"], "");
  EXPECT_EQ((*state)["model_sha256"], "");
  EXPECT_EQ((*state)["turns"], 0);
  EXPECT_EQ((*state)["chat_over"], false);
  model_debugger_runner_delete(runner);
}

TEST(RunnerCTest, StateSaysHowMuchRoomItNeeds) {
  char error[256] = "";
  ModelDebuggerRunner* runner =
      model_debugger_runner_start("127.0.0.1:0", error, sizeof(error));
  ASSERT_NE(runner, nullptr) << error;
  char small[16];
  const size_t needed =
      model_debugger_runner_get_state(runner, small, sizeof(small));
  EXPECT_GT(needed, sizeof(small));
  EXPECT_EQ(std::string(small).size(), sizeof(small) - 1);
  EXPECT_EQ(model_debugger_runner_get_state(runner, nullptr, 0), needed);
  model_debugger_runner_delete(runner);
}

TEST(RunnerCTest, DeleteTakesNull) { model_debugger_runner_delete(nullptr); }

}  // namespace
}  // namespace model_debugger::runner
