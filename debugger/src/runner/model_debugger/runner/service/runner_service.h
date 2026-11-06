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
// Runner. One model for the service's whole life, and one chat at a time.

#ifndef MODEL_DEBUGGER_RUNNER_SERVICE_RUNNER_SERVICE_H_
#define MODEL_DEBUGGER_RUNNER_SERVICE_RUNNER_SERVICE_H_

#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

#include "grpcpp/grpcpp.h"
#include "model_debugger/protocol/runner.grpc.pb.h"
#include "model_debugger/protocol/runner.pb.h"
#include "model_debugger/runner/llm/runtime.h"
#include "model_debugger/runner/service/model_store.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace model_debugger::runner::service {

// What the service is doing now: what a window shows of a runner.
struct State {
  protocol::HostInfo host;   // The machine, as GetInfo reports it.
  std::string client;        // The latest caller's peer; empty until one calls.
  std::string model_sha256;  // The loaded model; empty until Load.
  int turns = 0;             // The chat's completed turns, released or not.
  bool chat_over = false;    // A Send was cancelled or failed; nothing more is
                             // generated until a CreateChat.
};

// Implements model_debugger.protocol.Runner. Thread-safe: gRPC calls it from
// its own threads; the runtime underneath is driven from one at a time.
class RunnerService final : public protocol::Runner::Service {
 public:
  // Makes the service on `runtime`, which serves every call. The host is
  // measured once, for GetInfo. Everything lives in this platform's
  // CacheDirectory(): the model store `models/` (a ModelStore, which every
  // Runner of this cache directory shares), and `run.<pid>/`, where the
  // runtime writes and each turn's capture is kept as `turn-<n>/` until
  // released, removed with the service. Run directories of processes that are
  // gone are removed, and a leftover one of this process's id is made anew.
  // FailedPrecondition if the environment names no cache directory or the
  // model store or the run directory cannot be created.
  static absl::StatusOr<std::unique_ptr<RunnerService>> Create(
      std::unique_ptr<llm::Runtime> runtime);

  ~RunnerService() override;

  // Who this runner is. Never fails.
  grpc::Status GetInfo(grpc::ServerContext* context,
                       const protocol::GetInfoRequest* request,
                       protocol::RunnerInfo* response) override;

  // Puts a model in the store, written to a name of its own and kept once
  // the bytes hash to the header's sha256. Once the header is read, the
  // store keeps only this model (ModelStore::KeepOnly), and the service holds
  // it until it names another or goes, so that no Runner sharing the store
  // deletes it before the Load. Already in the store: OK without reading the
  // stream. DATA_LOSS if the hash disagrees or fewer bytes arrive than
  // announced, INVALID_ARGUMENT for a stream that does not start with a
  // header or carries more bytes than announced, RESOURCE_EXHAUSTED if the
  // disk is full, INTERNAL if writing fails otherwise. Nothing is kept on
  // failure.
  grpc::Status UploadModel(
      grpc::ServerContext* context,
      grpc::ServerReader<protocol::UploadModelRequest>* reader,
      protocol::UploadModelResponse* response) override;

  // Loads the one model, without blocking GetState meanwhile. Refused with
  // FAILED_PRECONDITION, changing nothing, if one is loaded, a Load or
  // CreateChat is in progress, or a Load failed before. Then the store keeps
  // only request->sha256 (ModelStore::KeepOnly), and the loaded model is held
  // for the service's life. NOT_FOUND, no failure, only if the store has no
  // file of request->sha256. Before the store is looked in or changed:
  // INVALID_ARGUMENT for a bad sha256, a context_length
  // that is not positive, a device that is not CPU or GPU, or runtime options
  // the runtime's load_options_schema refuses; for the GPU, what the
  // runtime's CanRunOnGpu says if not OK. CANCELLED if the call was cancelled
  // by the time the runtime returned, the model then discarded; else what the
  // runtime says, its NotFound as INTERNAL. Any failure is final: every Load
  // after it is refused.
  grpc::Status Load(grpc::ServerContext* context,
                    const protocol::LoadRequest* request,
                    protocol::LoadResponse* response) override;

  // Creates the chat, without blocking GetState meanwhile. Refused with
  // FAILED_PRECONDITION, changing nothing, if no model is loaded, or a Load,
  // CreateChat or Send is in progress. Otherwise the chat there is, live or
  // over, is deleted before the new one is created, so that two contexts
  // never take memory at once, with its turns' captures and what it left in
  // its capture directory; the new chat's turns count from 1.
  // INVALID_ARGUMENT for a fixed option out of the protocol's range or
  // runtime options the runtime's chat_options_schema refuses; CANCELLED if
  // the call was cancelled by the time the runtime returned, the chat then
  // discarded; else what the runtime says, its RESOURCE_EXHAUSTED for
  // starting messages the context has no room for with the trailing metadata
  // of Send's. A failure leaves no chat, and a CreateChat after it may
  // succeed.
  grpc::Status CreateChat(grpc::ServerContext* context,
                          const protocol::CreateChatRequest* request,
                          protocol::CreateChatResponse* response) override;

  // Streams the reply to one turn, chunks then the completion, and files the
  // turn's capture as `<data_dir>/turn-<n>/`. Refused with
  // FAILED_PRECONDITION, changing nothing, if there is no chat, the chat is
  // over, or a Send is in progress. A cancelled call cancels the chat within
  // tens of milliseconds, even before the first chunk, and ends with
  // CANCELLED, never as a filed turn; a cancel that comes while the prompt
  // is still being prepared takes effect only at the first chunk, once the
  // prompt is read. A cancel that comes after the model finished, until the
  // completion is written, ends the same way; one that comes later finds the
  // turn done, and the turn counts. RESOURCE_EXHAUSTED, before any chunk,
  // refuses a turn that needs more tokens than the context has left, its
  // trailing metadata "turn-tokens" and "tokens-left" saying how many. Any
  // failure but INVALID_ARGUMENT and RESOURCE_EXHAUSTED, a cancellation
  // included, ends the chat.
  grpc::Status Send(
      grpc::ServerContext* context, const protocol::SendRequest* request,
      grpc::ServerWriter<protocol::SendResponse>* writer) override;

  // Streams the files of a completed turn by name: for each its path and
  // size, its bytes 1 MiB per message, hashed as they go, then its SHA-256.
  // NOT_FOUND for a turn that has none, or that is released or whose chat is
  // closed meanwhile.
  // INTERNAL if the turn's directory holds anything but regular files, before
  // any file is sent; and INTERNAL if a file cannot be read, after the files
  // sent so far.
  grpc::Status ReadCaptures(
      grpc::ServerContext* context,
      const protocol::ReadCapturesRequest* request,
      grpc::ServerWriter<protocol::CaptureChunk>* writer) override;

  // Deletes a turn's capture; NOT_FOUND for a turn that has none.
  grpc::Status ReleaseCaptures(
      grpc::ServerContext* context,
      const protocol::ReleaseCapturesRequest* request,
      protocol::ReleaseCapturesResponse* response) override;

  // A copy of the state now. Callable from any thread, at any time.
  State GetState();

 private:
  // The service on `runtime`, with the model store and the run directory
  // Create has made.
  RunnerService(std::unique_ptr<llm::Runtime> runtime,
                std::filesystem::path model_store,
                std::filesystem::path data_dir);

  // Notes the peer of the latest call, which the runner's window shows.
  void NoteClient(const grpc::ServerContext& context);

  // Names the model of `sha256` as the one this service uses next: lets go of
  // the one named before, unless loaded, and has the store keep only this
  // one and the ones Runners hold. Called without mutex_ held.
  void NameModel(std::string_view sha256);

  // The directory of a completed turn's capture; NotFound for a turn that has
  // none: not completed, cancelled, or released.
  absl::StatusOr<std::filesystem::path> TurnDirectory(int turn) const;

  const std::unique_ptr<llm::Runtime> runtime_;
  // runtime_->Info(), read once: GetInfo must not call the runtime while a
  // Load drives it.
  const protocol::RuntimeInfo runtime_info_;
  // runtime_->CanRunOnGpu(), read once: the host reports a GPU if it is OK,
  // and a Load on the GPU is refused with it if not.
  const absl::Status gpu_;
  const protocol::HostInfo host_;
  ModelStore model_store_;
  const std::filesystem::path data_dir_;

  std::mutex mutex_;  // NOLINT(build/c++11)
  std::string client_;  // context.peer() of the latest call; empty until one.
  // The model last uploaded, held from the upload until another is named
  // or the service goes: the Load that follows, here or on another Runner
  // sharing the store, finds it.
  std::optional<ModelStore::Hold> uploaded_;
  std::unique_ptr<llm::Model> model_;
  std::string model_sha256_;                // Set with model_.
  std::optional<ModelStore::Hold> loaded_;  // model_'s file; set with model_.
  std::unique_ptr<llm::Chat> chat_;
  bool loading_ = false;      // A Load or CreateChat is in progress.
  bool load_failed_ = false;  // A Load failed: no other is made.
  bool sending_ = false;
  bool chat_over_ = false;
  // The chat's completed turns; each has a directory until released or the
  // chat is closed.
  int turns_ = 0;
  std::filesystem::path capture_dir_;  // The chat's, once a turn told it.
  // The CreateChats begun, each closing the chat before: a ReadCaptures that
  // sees the count change finds its turn gone, though a later chat may have
  // filed a turn of its number.
  int chats_ = 0;
};

}  // namespace model_debugger::runner::service

#endif  // MODEL_DEBUGGER_RUNNER_SERVICE_RUNNER_SERVICE_H_
