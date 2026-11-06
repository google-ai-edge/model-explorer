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

// Asks a Runner who it is: calls GetInfo and prints the RunnerInfo as JSON.

#include <chrono>  // NOLINT(build/c++11)
#include <iostream>
#include <memory>
#include <string>

#include "grpcpp/grpcpp.h"
#include "model_debugger/protocol/runner.grpc.pb.h"
#include "model_debugger/protocol/runner.pb.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/status/status.h"
#include "google/protobuf/json/json.h"

ABSL_FLAG(std::string, address, "127.0.0.1:8769",
          "HOST:PORT of the Runner to ask.");
ABSL_FLAG(int, timeout_seconds, 10, "How long to wait for the answer.");

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage("Prints a Model Debugger runner's GetInfo.");
  absl::ParseCommandLine(argc, argv);
  std::unique_ptr<model_debugger::protocol::Runner::Stub> stub =
      model_debugger::protocol::Runner::NewStub(grpc::CreateChannel(
          absl::GetFlag(FLAGS_address),
          grpc::InsecureChannelCredentials()));
  grpc::ClientContext context;
  context.set_deadline(
      std::chrono::system_clock::now() +
      std::chrono::seconds(absl::GetFlag(FLAGS_timeout_seconds)));
  model_debugger::protocol::RunnerInfo info;
  if (grpc::Status status = stub->GetInfo(
          &context, model_debugger::protocol::GetInfoRequest(), &info);
      !status.ok()) {
    std::cerr << "get_info: " << status.error_message() << "\n";
    return 1;
  }
  std::string json;
  if (absl::Status printed =
          grpc::protobuf::json::MessageToJsonString(info, &json);
      !printed.ok()) {
    std::cerr << "get_info: " << printed.message() << "\n";
    return 1;
  }
  std::cout << json << "\n";
  return 0;
}
