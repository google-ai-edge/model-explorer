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

// A Runner starts, answers, stops: no model is needed for any of it.

#include "model_debugger/runner/runner.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <filesystem>
#include <memory>
#include <string>

#include "grpcpp/grpcpp.h"
#include "model_debugger/runner/testing/test_utils.h"
#include "model_debugger/v1/runner.grpc.pb.h"
#include "model_debugger/v1/runner.pb.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace model_debugger::runner {
namespace {

const testing::Environment* const kTestHome =
    testing::AddGlobalTestEnvironment(new TestHome);

constexpr char kLoopback[] = "127.0.0.1:0";

TEST(RunnerTest, StartRefusesAnAddressItCannotBind) {
  EXPECT_EQ(Runner::Start("").status().code(), absl::StatusCode::kUnavailable);
}

TEST(RunnerTest, StartRefusesAPortAnotherProcessListensOn) {
  // Another Runner's socket, as gRPC makes it by default: it allows the port
  // to be reused, so only a Runner that does not ask for reuse is refused.
  const int other = socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(other, 0);
  const int on = 1;
  ASSERT_EQ(setsockopt(other, SOL_SOCKET, SO_REUSEPORT, &on, sizeof(on)), 0);
  sockaddr_in address = {};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  socklen_t size = sizeof(address);
  ASSERT_EQ(bind(other, reinterpret_cast<sockaddr*>(&address), size), 0);
  ASSERT_EQ(listen(other, 1), 0);
  ASSERT_EQ(getsockname(other, reinterpret_cast<sockaddr*>(&address), &size),
            0);
  const int port = ntohs(address.sin_port);
  EXPECT_EQ(Runner::Start("127.0.0.1:" + std::to_string(port)).status().code(),
            absl::StatusCode::kUnavailable);
  close(other);
}

TEST(RunnerTest, StartsOnAFreePortAndReportsIt) {
  absl::StatusOr<std::unique_ptr<Runner>> runner = Runner::Start(kLoopback);
  ASSERT_TRUE(runner.ok()) << runner.status();
  const State state = (*runner)->GetState();
  EXPECT_EQ(state.address, "127.0.0.1");
  EXPECT_NE(state.port, 0);
  EXPECT_EQ(state.service.host.platform(), kPlatform);
  EXPECT_TRUE(state.service.client.empty());
  EXPECT_TRUE(state.service.model_sha256.empty());
  EXPECT_EQ(state.service.turns, 0);
}

TEST(RunnerTest, ServesTheProtocolAndSeesItsClient) {
  absl::StatusOr<std::unique_ptr<Runner>> runner = Runner::Start(kLoopback);
  ASSERT_TRUE(runner.ok()) << runner.status();
  std::unique_ptr<v1::Runner::Stub> stub =
      v1::Runner::NewStub(grpc::CreateChannel(
          "127.0.0.1:" + std::to_string((*runner)->GetState().port),
          grpc::InsecureChannelCredentials()));
  grpc::ClientContext context;
  v1::RunnerInfo info;
  ASSERT_TRUE(stub->GetInfo(&context, v1::GetInfoRequest(), &info).ok());
  EXPECT_EQ(info.protocol_version(), "1.0.0");
  EXPECT_EQ(info.runtime().version().size(), 40u) << "the LiteRT-LM commit";
  EXPECT_FALSE((*runner)->GetState().service.client.empty());
}

TEST(RunnerTest, ItsRunDirectoryLivesWithIt) {
  const std::filesystem::path run =
      UseTestCacheDirectory() / ("run." + std::to_string(getpid()));
  {
    absl::StatusOr<std::unique_ptr<Runner>> runner = Runner::Start(kLoopback);
    ASSERT_TRUE(runner.ok()) << runner.status();
    EXPECT_TRUE(std::filesystem::is_directory(run));
  }
  EXPECT_FALSE(std::filesystem::exists(run));
}

TEST(RunnerTest, ItRemovesTheRunsOfProcessesThatAreGone) {
  const std::filesystem::path root = UseTestCacheDirectory();
  // No process has this id: pids stay far below it.
  const std::filesystem::path orphan = root / "run.2147483647";
  std::filesystem::create_directories(orphan / "turn-1");
  const std::filesystem::path models = root / "models";
  std::filesystem::create_directories(models);
  absl::StatusOr<std::unique_ptr<Runner>> runner = Runner::Start(kLoopback);
  ASSERT_TRUE(runner.ok()) << runner.status();
  EXPECT_FALSE(std::filesystem::exists(orphan));
  EXPECT_TRUE(std::filesystem::is_directory(models)) << "models are kept";
}

}  // namespace
}  // namespace model_debugger::runner
