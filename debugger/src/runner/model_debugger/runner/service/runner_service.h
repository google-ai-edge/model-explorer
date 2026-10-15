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

// The protocol's Runner service on one llm::Runtime: the gRPC side of the
// Runner. One client for the service's whole life, one model, one chat.

#ifndef MODEL_DEBUGGER_RUNNER_SERVICE_RUNNER_SERVICE_H_
#define MODEL_DEBUGGER_RUNNER_SERVICE_RUNNER_SERVICE_H_

#include <filesystem>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/grpcpp.h"
#include "model_debugger/runner/llm/runtime.h"
#include "model_debugger/v1/runner.grpc.pb.h"
#include "model_debugger/v1/runner.pb.h"

namespace model_debugger::runner::service {

// What the service is doing now: what a window shows of a runner.
struct State {
  v1::HostInfo host;         // The machine, as GetInfo reports it.
  std::string client;        // The one client's peer; empty until it calls.
  std::string model_sha256;  // The loaded model; empty until Load.
  int turns = 0;             // Completed turns, released or not.
  bool chat_over = false;    // A Send was cancelled; nothing more is generated.
};

// Implements model_debugger.v1.Runner. Thread-safe: gRPC calls it from its
// own threads; the runtime underneath is driven from one at a time.
class RunnerService final : public v1::Runner::Service {
 public:
  // Makes the service on `runtime`, which serves every call. The host is
  // measured once, for GetInfo. Everything lives in this platform's
  // CacheDirectory(): model files by sha256 as `models/<sha256>.litertlm`,
  // and `run.<pid>/`, where the runtime writes and each turn's capture is
  // kept as `turn-<n>/` until released, removed with the service. Run
  // directories of processes that are gone are removed, and a leftover one of
  // this process's id is made anew. FailedPrecondition if the environment
  // names no cache directory or the model store or the run directory cannot
  // be created.
  static absl::StatusOr<std::unique_ptr<RunnerService>> Create(
      std::unique_ptr<llm::Runtime> runtime);

  ~RunnerService() override;

  // Who this runner is. Never fails.
  grpc::Status GetInfo(grpc::ServerContext* context,
                       const v1::GetInfoRequest* request,
                       v1::RunnerInfo* response) override;

  // Puts a model in the store as `<model_store>/<sha256>.litertlm`, written
  // to a temporary name and renamed once the bytes hash to the header's
  // sha256. Already in the store: OK without reading the stream. DATA_LOSS
  // if the hash disagrees or fewer bytes arrive than announced,
  // INVALID_ARGUMENT for a stream that does not start with a header or
  // carries more bytes than announced, RESOURCE_EXHAUSTED if the disk is
  // full, INTERNAL if writing fails otherwise. Nothing is kept on failure.
  grpc::Status UploadModel(grpc::ServerContext* context,
                           grpc::ServerReader<v1::UploadModelRequest>* reader,
                           v1::UploadModelResponse* response) override;

  // Loads the one model, without blocking GetState meanwhile:
  // FAILED_PRECONDITION if one is loaded or a Load or CreateChat is in
  // progress, NOT_FOUND only if the store has no file of request->sha256,
  // else what the runtime says, its NotFound as INTERNAL.
  grpc::Status Load(grpc::ServerContext* context,
                    const v1::LoadRequest* request,
                    v1::LoadResponse* response) override;

  // Creates the one chat, without blocking GetState meanwhile:
  // FAILED_PRECONDITION if no model is loaded, a chat exists, or a Load or
  // CreateChat is in progress, else what the runtime says.
  grpc::Status CreateChat(grpc::ServerContext* context,
                          const v1::CreateChatRequest* request,
                          v1::CreateChatResponse* response) override;

  // Streams the reply to one turn, chunks then the completion, and files the
  // turn's capture as `<data_dir>/turn-<n>/`. FAILED_PRECONDITION if there is
  // no chat, the chat is over, or a Send is in progress. A cancelled call
  // cancels the chat within tens of milliseconds, even before the first
  // chunk, and ends with CANCELLED, never as a filed turn. Any failure but
  // INVALID_ARGUMENT, a cancellation included, ends the chat.
  grpc::Status Send(grpc::ServerContext* context,
                    const v1::SendRequest* request,
                    grpc::ServerWriter<v1::SendResponse>* writer) override;

  // The files of a completed turn; NOT_FOUND for a turn that has none.
  grpc::Status ListCaptures(grpc::ServerContext* context,
                            const v1::ListCapturesRequest* request,
                            v1::ListCapturesResponse* response) override;

  // The bytes of one captured file, 1 MiB per message; NOT_FOUND if there is
  // no such file in that turn.
  grpc::Status ReadCapture(
      grpc::ServerContext* context, const v1::ReadCaptureRequest* request,
      grpc::ServerWriter<v1::CaptureData>* writer) override;

  // Deletes a turn's capture; NOT_FOUND for a turn that has none.
  grpc::Status ReleaseCaptures(grpc::ServerContext* context,
                               const v1::ReleaseCapturesRequest* request,
                               v1::ReleaseCapturesResponse* response) override;

  // A copy of the state now. Callable from any thread, at any time.
  State GetState();

 private:
  // The service on `runtime`, with the model store and the run directory
  // Create has made.
  RunnerService(std::unique_ptr<llm::Runtime> runtime,
                std::filesystem::path model_store,
                std::filesystem::path data_dir);

  // The first peer to call is the one client; every other peer gets
  // FAILED_PRECONDITION until the service is destroyed.
  grpc::Status Admit(const grpc::ServerContext& context)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // The directory of a completed turn's capture; NotFound for a turn that has
  // none: not completed, cancelled, or released.
  absl::StatusOr<std::filesystem::path> TurnDirectory(int turn) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  const std::unique_ptr<llm::Runtime> runtime_;
  // runtime_->Info(), read once: GetInfo must not call the runtime while a
  // Load drives it.
  const v1::RuntimeInfo runtime_info_;
  const v1::HostInfo host_;
  const std::filesystem::path model_store_;
  const std::filesystem::path data_dir_;

  mutable absl::Mutex mutex_;
  std::string client_ ABSL_GUARDED_BY(
      mutex_);  // context.peer() of the one client; empty until then.
  std::unique_ptr<llm::Model> model_ ABSL_GUARDED_BY(mutex_);
  std::string model_sha256_ ABSL_GUARDED_BY(mutex_);  // Set with model_.
  std::unique_ptr<llm::Chat> chat_ ABSL_GUARDED_BY(mutex_);
  bool loading_ ABSL_GUARDED_BY(mutex_) =
      false;  // A Load or CreateChat is in progress.
  bool sending_ ABSL_GUARDED_BY(mutex_) = false;
  bool chat_over_ ABSL_GUARDED_BY(mutex_) = false;
  int turns_ ABSL_GUARDED_BY(mutex_) =
      0;  // Completed turns; each has a directory until released.
  std::filesystem::path capture_dir_
      ABSL_GUARDED_BY(mutex_);  // The chat's, once a turn told it.
};

}  // namespace model_debugger::runner::service

#endif  // MODEL_DEBUGGER_RUNNER_SERVICE_RUNNER_SERVICE_H_
