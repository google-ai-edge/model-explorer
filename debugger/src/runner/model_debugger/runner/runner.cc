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

#include "model_debugger/runner/runner.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/grpcpp.h"
#include "model_debugger/runner/llm/litert_lm/litert_lm_runtime.h"
#include "model_debugger/runner/service/runner_service.h"

namespace model_debugger::runner {
namespace {

// How long the destructor lets the calls in flight finish before it cancels
// them: a Send notices within tens of milliseconds, a Load not before it ends.
constexpr absl::Duration kShutdownGrace = absl::Seconds(3);

}  // namespace

absl::StatusOr<std::unique_ptr<Runner>> Runner::Start(std::string listen) {
  absl::StatusOr<std::unique_ptr<service::RunnerService>> service =
      service::RunnerService::Create(std::make_unique<llm::LiteRtLmRuntime>());
  if (!service.ok()) return service.status();
  grpc::ServerBuilder builder;
  // A port another Runner holds must fail here, not be shared with it.
  builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);
  int port = 0;
  builder.AddListeningPort(listen, grpc::InsecureServerCredentials(), &port);
  builder.RegisterService(service->get());
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  if (server == nullptr || port == 0) {
    return absl::UnavailableError(
        absl::StrCat("Cannot listen on ", listen, "."));
  }
  return std::unique_ptr<Runner>(new Runner(
      std::move(listen), *std::move(service), std::move(server), port));
}

Runner::Runner(std::string listen,
               std::unique_ptr<service::RunnerService> service,
               std::unique_ptr<grpc::Server> server, int port)
    : listen_(std::move(listen)),
      port_(port),
      service_(std::move(service)),
      server_(std::move(server)) {}

Runner::~Runner() {
  server_->Shutdown(absl::ToChronoTime(absl::Now() + kShutdownGrace));
  server_.reset();  // Before the service it calls.
}

State Runner::GetState() {
  State state;
  state.address = listen_.substr(0, listen_.rfind(':'));
  state.port = port_;
  state.service = service_->GetState();
  return state;
}

void Runner::Wait() { server_->Wait(); }

}  // namespace model_debugger::runner
