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

// A Runner: the protocol's service on LiteRT-LM, listening. The facade over
// the layers; every launch form shares it. The command line (cli/) and the
// Apps on macOS, iOS and Android (app/, through runner_c.h) assemble nothing
// themselves: they start this and show what it reports.

#ifndef MODEL_DEBUGGER_RUNNER_RUNNER_H_
#define MODEL_DEBUGGER_RUNNER_RUNNER_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "grpcpp/server.h"
#include "model_debugger/runner/service/runner_service.h"

namespace model_debugger::runner {

// What a Runner shows of itself: a window draws this and nothing else.
struct State {
  std::string address;  // The host part of `listen`.
  int port = 0;         // The port actually bound.
  service::State service;
};

// The protocol's service on LiteRT-LM, listening on one address from Start
// until destroyed. Thread-safe: GetState may be called from any thread.
class Runner {
 public:
  // Starts serving on `listen`, HOST:PORT; port 0 picks a free one, which
  // State reports. Fails with RunnerService::Create's reason if the service
  // cannot be made, and with Unavailable if the address cannot be bound, a
  // port another process listens on included.
  static absl::StatusOr<std::unique_ptr<Runner>> Start(std::string listen);

  // Stops accepting calls and waits a few seconds for the in-flight ones,
  // then cancels those still running and waits for them to return.
  ~Runner();

  Runner(const Runner&) = delete;
  Runner& operator=(const Runner&) = delete;

  // A copy of the state now. Callable from any thread.
  State GetState();

  // Blocks for as long as the Runner serves: until it is destroyed.
  void Wait();

 private:
  Runner(std::string listen, std::unique_ptr<service::RunnerService> service,
         std::unique_ptr<grpc::Server> server, int port);

  const std::string listen_;
  const int port_;
  // Declared so that the server, which calls the service, is destroyed first.
  std::unique_ptr<service::RunnerService> service_;
  std::unique_ptr<grpc::Server> server_;
};

}  // namespace model_debugger::runner

#endif  // MODEL_DEBUGGER_RUNNER_RUNNER_H_
