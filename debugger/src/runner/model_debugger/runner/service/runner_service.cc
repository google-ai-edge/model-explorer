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

#include "model_debugger/runner/service/runner_service.h"

#include <signal.h>
#include <unistd.h>

// NOLINTBEGIN(build/c++11)
// NOLINTBEGIN(build/c++17)
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

#include "grpcpp/grpcpp.h"
#include "model_debugger/protocol/runner.grpc.pb.h"
#include "model_debugger/protocol/runner.pb.h"
#include "model_debugger/runner/llm/runtime.h"
#include "model_debugger/runner/service/cache_directory.h"
#include "model_debugger/runner/service/host_info.h"
#include "model_debugger/runner/service/model_store.h"
#include "model_debugger/runner/service/options_schema.h"
#include "model_debugger/runner/util/disk_space.h"
#include "openssl/sha.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"

namespace model_debugger::runner::service {
namespace {

// The file the runtime appends a line to per decode step; the turn's lines are
// copied to the turn's directory.
constexpr std::string_view kTokensFile = "generated_tokens.jsonl";

// How many bytes of a captured file one CaptureChunk carries.
constexpr std::streamsize kCaptureChunkBytes = 1 << 20;

// How often a Send checks whether its client has cancelled.
constexpr std::chrono::milliseconds kCancelPollInterval(20);

// The refusal of a Load or CreateChat while another is in progress.
constexpr std::string_view kPreparing = "A Load or CreateChat is in progress.";

// The refusal of a Load after one failed.
constexpr std::string_view kLoadFailed =
    "A Load failed; a runner loads one model, once.";

// The refusal of a Send, or a CreateChat, while a Send is in progress.
constexpr std::string_view kSending = "A Send is in progress.";

// The trailing metadata of a Send or CreateChat refused for what the context
// has no room for, by the payload of the runtime's refusal it carries.
constexpr std::pair<std::string_view, std::string_view> kNoRoomMetadata[] = {
    {llm::kTurnTokensPayload, "turn-tokens"},
    {llm::kTokensLeftPayload, "tokens-left"},
};

// Gives `context` the trailing metadata of `refused`, a refusal for no room.
void AddNoRoomMetadata(const absl::Status& refused,
                       grpc::ServerContext& context) {
  for (const auto& [payload, key] : kNoRoomMetadata) {
    if (std::optional<absl::Cord> value = refused.GetPayload(payload);
        value.has_value()) {
      context.AddTrailingMetadata(std::string(key), std::string(*value));
    }
  }
}

// gRPC's codes are absl's canonical codes.
grpc::Status ToGrpc(const absl::Status& status) {
  return grpc::Status(static_cast<grpc::StatusCode>(status.code()),
                      std::string(status.message()));
}

bool IsSha256(std::string_view text) {
  if (text.size() != 64) return false;
  for (const char c : text) {
    if (!(('0' <= c && c <= '9') || ('a' <= c && c <= 'f'))) return false;
  }
  return true;
}

// Returns the names of the entries directly in `dir`, files or not.
std::set<std::string> EntryNames(const std::filesystem::path& dir) {
  std::set<std::string> names;
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator(dir, error)) {
    names.insert(entry.path().filename());
  }
  return names;
}

// Returns the names of the files of the capture in `turn_dir`, in order:
// every entry, each a regular file. Internal, naming the entry, for one that
// is not: a capture is flat, and a subdirectory would be dropped unseen.
absl::StatusOr<std::set<std::string>> CaptureFiles(
    const std::filesystem::path& turn_dir) {
  std::set<std::string> names;
  std::error_code error;
  for (const auto& entry :
       std::filesystem::directory_iterator(turn_dir, error)) {
    const std::filesystem::file_status status = entry.symlink_status(error);
    if (error) break;
    if (status.type() != std::filesystem::file_type::regular) {
      return absl::InternalError(absl::StrCat(
          entry.path().string(),
          " is not a regular file; a capture is the files directly in the "
          "turn's directory."));
    }
    names.insert(entry.path().filename());
  }
  if (error) {
    return absl::InternalError(
        absl::StrCat("Cannot list ", turn_dir.string(), ": ", error.message()));
  }
  return names;
}

// Returns the number of lines in `file`, 0 if it is not there.
int CountLines(const std::filesystem::path& file) {
  std::ifstream in(file);
  int lines = 0;
  std::string line;
  while (std::getline(in, line)) ++lines;
  return lines;
}

// A SHA-256 as bytes arrive.
class Sha256 {
 public:
  Sha256() { SHA256_Init(&context_); }
  void Update(std::string_view bytes) {
    SHA256_Update(&context_, bytes.data(), bytes.size());
  }
  // The digest, lower-case hexadecimal. Ends the computation.
  std::string Finish() {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256_Final(digest, &context_);
    return absl::BytesToHexString(std::string_view(
        reinterpret_cast<const char*>(digest), sizeof(digest)));
  }

 private:
  SHA256_CTX context_;
};

// Moves what the turn wrote into its own directory: the entries of
// `capture_dir` that were not there before, files or not, and the lines the
// tokens file gained. The runtime closes each tensor file after writing it and
// appends to the tokens file per line, so nothing is moved from under it. A
// write its disk has no room for ends the Runner (ExitIfOutOfRoom).
absl::Status FileTurn(const std::filesystem::path& capture_dir,
                      const std::set<std::string>& before, int lines_before,
                      const std::filesystem::path& turn_dir) {
  std::error_code error;
  std::filesystem::create_directories(turn_dir, error);
  if (error) {
    ExitIfOutOfRoom(turn_dir.string(), 1);
    return absl::InternalError(absl::StrCat("Cannot create ", turn_dir.string(),
                                            ": ", error.message()));
  }
  for (const std::string& name : EntryNames(capture_dir)) {
    if (name == kTokensFile || before.count(name) != 0) continue;
    std::filesystem::rename(capture_dir / name, turn_dir / name, error);
    if (error) {
      return absl::InternalError(
          absl::StrCat("Cannot move ", name, ": ", error.message()));
    }
  }
  std::ifstream in(capture_dir / kTokensFile);
  std::string lines;
  std::string line;
  for (int i = 0; std::getline(in, line); ++i) {
    if (i >= lines_before) absl::StrAppend(&lines, line, "\n");
  }
  if (in.bad()) {
    return absl::InternalError(
        absl::StrCat("Cannot read the turn's ", kTokensFile, "."));
  }
  const std::filesystem::path tokens = turn_dir / kTokensFile;
  std::ofstream out(tokens);
  out << lines;
  out.close();
  if (out.fail()) {
    const std::uintmax_t held = std::min<std::uintmax_t>(
        std::filesystem::file_size(tokens, error), lines.size());
    ExitIfOutOfRoom(
        tokens.string(),
        error ? 1 : std::max<std::uintmax_t>(lines.size() - held, 1));
    return absl::InternalError(
        absl::StrCat("Cannot copy the turn's ", kTokensFile, "."));
  }
  return absl::OkStatus();
}

// Checks the fixed chat options against the ranges the protocol gives them,
// which hold for every runtime.
absl::Status CheckChatOptions(const protocol::ChatOptions& options) {
  if (options.max_completion_tokens() < 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ChatOptions.max_completion_tokens is ",
        options.max_completion_tokens(), "; it must be at least 1."));
  }
  // Written so that NaN, which compares false, is refused too.
  if (options.has_temperature() && !(options.temperature() >= 0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("ChatOptions.temperature is ", options.temperature(),
                     "; it must be at least 0."));
  }
  if (options.has_top_p() && !(options.top_p() >= 0 && options.top_p() <= 1)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ChatOptions.top_p is ", options.top_p(), "; it must be from 0 to 1."));
  }
  if (options.has_top_k() && options.top_k() < 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ChatOptions.top_k is ", options.top_k(), "; it must be at least 1."));
  }
  return absl::OkStatus();
}

// Cancels a chat once the client cancels its call, checking every
// kCancelPollInterval until destroyed. The chunk callback notices only when a
// chunk arrives, and none arrives while the model reads the prompt.
class CancelWatcher {
 public:
  CancelWatcher(grpc::ServerContext& context, llm::Chat& chat)
      : thread_([this, &context, &chat] { Watch(context, chat); }) {}

  CancelWatcher(const CancelWatcher&) = delete;
  CancelWatcher& operator=(const CancelWatcher&) = delete;

  ~CancelWatcher() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopped_ = true;
    }
    stop_.notify_one();
    thread_.join();
  }

 private:
  // Runs on thread_ until the client cancels or the watcher is destroyed.
  void Watch(grpc::ServerContext& context, llm::Chat& chat) {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop_.wait_for(lock, kCancelPollInterval,
                           [this] { return stopped_; })) {
      if (context.IsCancelled()) {
        chat.Cancel();
        return;
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable stop_;  // Signalled when stopped_ is set.
  bool stopped_ = false;
  std::thread thread_;  // Last: it starts once the members it uses exist.
};

// The prefix of a directory a running service owns in the cache root, the
// process id after it.
constexpr std::string_view kRunPrefix = "run.";

// Removes the run directories of processes that are gone: a Runner that
// crashed or was killed had no chance to remove its own.
void RemoveOrphanedRuns(const std::filesystem::path& root) {
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator(root, error)) {
    const std::string name = entry.path().filename();
    if (!absl::StartsWith(name, kRunPrefix)) continue;
    int pid = 0;
    if (!absl::SimpleAtoi(name.substr(kRunPrefix.size()), &pid) || pid <= 0) {
      continue;
    }
    // ESRCH: no such process. EPERM means it exists and is another's.
    if (kill(pid, 0) == -1 && errno == ESRCH) {
      std::filesystem::remove_all(entry.path(), error);
    }
  }
}

}  // namespace

absl::StatusOr<std::unique_ptr<RunnerService>> RunnerService::Create(
    std::unique_ptr<llm::Runtime> runtime) {
  const std::filesystem::path root = CacheDirectory();
  if (root.empty()) {
    return absl::FailedPreconditionError(
        "The environment names no cache directory: HOME is not set.");
  }
  RemoveOrphanedRuns(root);
  const std::filesystem::path model_store = root / "models";
  const std::filesystem::path data_dir =
      root / absl::StrCat(kRunPrefix, getpid());
  // An earlier process with this id may have left its directory behind.
  std::error_code error;
  std::filesystem::remove_all(data_dir, error);
  if (error) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Cannot remove ", data_dir.string(), ": ", error.message()));
  }
  for (const std::filesystem::path& dir : {model_store, data_dir}) {
    std::filesystem::create_directories(dir, error);
    if (error) {
      return absl::FailedPreconditionError(
          absl::StrCat("Cannot create ", dir.string(), ": ", error.message()));
    }
  }
  return std::unique_ptr<RunnerService>(
      new RunnerService(std::move(runtime), model_store, data_dir));
}

RunnerService::RunnerService(std::unique_ptr<llm::Runtime> runtime,
                             std::filesystem::path model_store,
                             std::filesystem::path data_dir)
    : runtime_(std::move(runtime)),
      runtime_info_(runtime_->Info()),
      gpu_(runtime_->CanRunOnGpu()),
      host_(MeasureHost(gpu_.ok())),
      model_store_(std::move(model_store)),
      data_dir_(std::move(data_dir)) {}

RunnerService::~RunnerService() {
  // The runtime, which writes there, goes first.
  chat_.reset();
  model_.reset();
  std::error_code error;
  std::filesystem::remove_all(data_dir_, error);
}

void RunnerService::NoteClient(const grpc::ServerContext& context) {
  // Called with mutex_ held.
  client_ = context.peer();
}

void RunnerService::NameModel(std::string_view sha256) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    uploaded_.reset();
  }
  model_store_.KeepOnly(sha256);
}

grpc::Status RunnerService::GetInfo(grpc::ServerContext* context,
                                    const protocol::GetInfoRequest* /*request*/,
                                    protocol::RunnerInfo* response) {
  std::lock_guard<std::mutex> lock(mutex_);
  NoteClient(*context);
  // The version of the protocol this Runner was built against, as the
  // proto's file option names it.
  response->set_protocol_version(
      protocol::RunnerInfo::descriptor()->file()->options().GetExtension(
          protocol::protocol_version));
  *response->mutable_host() = host_;
  *response->mutable_runtime() = runtime_info_;
  return grpc::Status::OK;
}

grpc::Status RunnerService::UploadModel(
    grpc::ServerContext* context,
    grpc::ServerReader<protocol::UploadModelRequest>* reader,
    protocol::UploadModelResponse* /*response*/) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    NoteClient(*context);
  }
  protocol::UploadModelRequest request;
  if (!reader->Read(&request) || !request.has_header()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "The stream does not start with a header.");
  }
  const protocol::ModelHeader header = request.header();
  if (!IsSha256(header.sha256())) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "sha256 is not 64 lower-case hexadecimal digits.");
  }
  if (header.size() < 0) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "size is negative.");
  }
  // Deleting first leaves the upload the room the other models took.
  NameModel(header.sha256());
  absl::StatusOr<ModelStore::Hold> held =
      model_store_.HoldModel(header.sha256());
  if (!held.ok() && !absl::IsNotFound(held.status())) {
    return ToGrpc(held.status());
  }
  if (!held.ok()) {
    absl::StatusOr<ModelStore::Upload> upload =
        model_store_.StartUpload(header.sha256());
    if (!upload.ok()) return ToGrpc(upload.status());
    // Kept only once the bytes prove to be the model: a reader of the store
    // never sees half a file. On any failure the upload is deleted with
    // `upload`.
    Sha256 sha256;
    int64_t received = 0;
    while (reader->Read(&request)) {
      if (!request.has_data()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "A second header arrived in the stream.");
      }
      const int64_t size = static_cast<int64_t>(request.data().size());
      if (size > header.size() - received) {
        return grpc::Status(
            grpc::StatusCode::INVALID_ARGUMENT,
            absl::StrCat("More than the ", header.size(),
                         " bytes the header announced arrived."));
      }
      if (absl::Status appended = upload->Append(request.data());
          !appended.ok()) {
        return ToGrpc(appended);
      }
      sha256.Update(request.data());
      received += size;
    }
    if (context->IsCancelled()) {
      return grpc::Status(grpc::StatusCode::CANCELLED,
                          "The upload was cancelled.");
    }
    if (received != header.size()) {
      return grpc::Status(grpc::StatusCode::DATA_LOSS,
                          absl::StrCat("The header announced ", header.size(),
                                       " bytes; ", received, " arrived."));
    }
    if (const std::string digest = sha256.Finish(); digest != header.sha256()) {
      return grpc::Status(grpc::StatusCode::DATA_LOSS,
                          absl::StrCat("The bytes hash to ", digest,
                                       ", not to ", header.sha256(), "."));
    }
    held = model_store_.Keep(*std::move(upload));
    if (!held.ok()) return ToGrpc(held.status());
  }
  std::lock_guard<std::mutex> lock(mutex_);
  uploaded_ = *std::move(held);
  return grpc::Status::OK;
}

grpc::Status RunnerService::Load(grpc::ServerContext* context,
                                 const protocol::LoadRequest* request,
                                 protocol::LoadResponse* response) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    NoteClient(*context);
    // Refusals for the state, which change nothing.
    if (model_ != nullptr) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          "A model is loaded; a runner loads one.");
    }
    if (loading_) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          std::string(kPreparing));
    }
    if (load_failed_) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          std::string(kLoadFailed));
    }
    // Any failure from here on but NOT_FOUND is final, until the Load
    // succeeds.
    load_failed_ = true;
    if (!IsSha256(request->sha256())) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "sha256 is not 64 lower-case hexadecimal digits.");
    }
    if (request->context_length() <= 0) {
      return grpc::Status(
          grpc::StatusCode::INVALID_ARGUMENT,
          absl::StrCat("LoadRequest.context_length is ",
                       request->context_length(), "; it must be positive."));
    }
    if (request->device() != protocol::CPU &&
        request->device() != protocol::GPU) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "LoadRequest.device is not CPU or GPU.");
    }
    // Before the store, as what follows: a Load that would be refused is
    // worth no upload.
    if (request->device() == protocol::GPU && !gpu_.ok()) return ToGrpc(gpu_);
    if (absl::Status valid = ValidateOptions(
            runtime_info_.load_options_schema(), request->runtime_options(),
            "LoadRequest.runtime_options");
        !valid.ok()) {
      return ToGrpc(valid);
    }
    loading_ = true;
  }
  // Without the lock, as the store waits for other Runners sharing it.
  NameModel(request->sha256());
  absl::StatusOr<ModelStore::Hold> held =
      model_store_.HoldModel(request->sha256());
  if (!held.ok()) {
    std::lock_guard<std::mutex> lock(mutex_);
    loading_ = false;
    if (!absl::IsNotFound(held.status())) return ToGrpc(held.status());
    // NOT_FOUND asks the client to upload the model, so it means only that,
    // and is no failure.
    load_failed_ = false;
    return grpc::Status(grpc::StatusCode::NOT_FOUND,
                        absl::StrCat("No model with sha256 ", request->sha256(),
                                     " in the model store."));
  }
  // Without the lock: loading takes seconds, and the window asks for the
  // state meanwhile. A cancel does not stop it; its result is discarded.
  absl::StatusOr<std::unique_ptr<llm::Model>> loaded = runtime_->Load(
      model_store_.ModelPath(request->sha256()), data_dir_, *request);
  // Declared after `loaded`, so released before it: a discarded model is
  // freed without the lock.
  std::lock_guard<std::mutex> lock(mutex_);
  loading_ = false;
  if (context->IsCancelled()) {
    return grpc::Status(grpc::StatusCode::CANCELLED,
                        "The Load was cancelled; the model is discarded.");
  }
  if (absl::IsNotFound(loaded.status())) {
    // The model is in the store; what the runtime misses is the runner's.
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        std::string(loaded.status().message()));
  }
  if (!loaded.ok()) return ToGrpc(loaded.status());
  load_failed_ = false;
  model_ = *std::move(loaded);
  model_sha256_ = request->sha256();
  loaded_ = *std::move(held);
  *response = model_->Info();
  return grpc::Status::OK;
}

grpc::Status RunnerService::CreateChat(
    grpc::ServerContext* context, const protocol::CreateChatRequest* request,
    protocol::CreateChatResponse* /*response*/) {
  llm::Model* model = nullptr;
  std::unique_ptr<llm::Chat> closed;
  int closed_turns = 0;
  std::filesystem::path closed_capture_dir;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    NoteClient(*context);
    // Refusals for the state, which change nothing.
    if (loading_) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          std::string(kPreparing));
    }
    if (model_ == nullptr) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          "No model is loaded.");
    }
    if (sending_) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          std::string(kSending));
    }
    // The chat there is, if any, is closed whatever follows: its turns are
    // gone at once, and their files below.
    closed = std::move(chat_);
    closed_turns = std::exchange(turns_, 0);
    closed_capture_dir = std::exchange(capture_dir_, {});
    chat_over_ = false;
    ++chats_;
    loading_ = true;
    model = model_.get();
  }
  // Without the lock, like Load: the model is kept until the service goes.
  // The old chat goes before the new one comes: a phone has no room for two
  // contexts. What it left in its capture directory, if a turn told where,
  // goes with its turns.
  closed.reset();
  std::error_code error;
  for (int turn = 1; turn <= closed_turns; ++turn) {
    std::filesystem::remove_all(data_dir_ / absl::StrCat("turn-", turn), error);
  }
  if (!closed_capture_dir.empty()) {
    std::filesystem::remove_all(closed_capture_dir, error);
  }
  absl::Status valid = CheckChatOptions(request->options());
  if (valid.ok()) {
    valid = ValidateOptions(runtime_info_.chat_options_schema(),
                            request->options().runtime_options(),
                            "ChatOptions.runtime_options");
  }
  absl::StatusOr<std::unique_ptr<llm::Chat>> chat =
      valid.ok() ? model->CreateChat(*request)
                 : absl::StatusOr<std::unique_ptr<llm::Chat>>(valid);
  // Discarded before the next CreateChat may begin, for the same reason.
  if (valid.ok() && context->IsCancelled()) {
    chat = absl::CancelledError(
        "The CreateChat was cancelled; the chat is discarded.");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  loading_ = false;
  if (!chat.ok()) {
    if (absl::IsResourceExhausted(chat.status())) {
      AddNoRoomMetadata(chat.status(), *context);
    }
    return ToGrpc(chat.status());
  }
  chat_ = *std::move(chat);
  return grpc::Status::OK;
}

grpc::Status RunnerService::Send(
    grpc::ServerContext* context, const protocol::SendRequest* request,
    grpc::ServerWriter<protocol::SendResponse>* writer) {
  llm::Chat* chat = nullptr;
  int turn_number = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    NoteClient(*context);
    // Refusals for the state, which change nothing.
    if (chat_ == nullptr) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          "No chat exists.");
    }
    if (chat_over_) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          "The chat has ended with a cancelled or failed "
                          "turn; a CreateChat starts another.");
    }
    if (sending_) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          std::string(kSending));
    }
    sending_ = true;
    chat = chat_.get();
    turn_number = turns_ + 1;
  }

  // What the capture directory holds before the turn, to tell the turn's
  // files apart afterwards. The directory is the chat's; the first turn
  // learns it from the result, so the snapshot is taken when known.
  std::set<std::string> before;
  int lines_before = 0;
  if (!capture_dir_.empty()) {
    before = EntryNames(capture_dir_);
    lines_before = CountLines(capture_dir_ / kTokensFile);
  }

  absl::StatusOr<llm::Turn> turn;
  {
    CancelWatcher watcher(*context, *chat);
    turn = chat->Send(request->message(), [&](const protocol::Chunk& chunk) {
      if (context->IsCancelled()) {
        chat->Cancel();
        return;
      }
      protocol::SendResponse response;
      *response.mutable_chunk() = chunk;
      writer->Write(response);
    });
  }

  std::unique_lock<std::mutex> lock(mutex_);
  if (turn.ok() && context->IsCancelled()) {
    // The model finished a turn the client is no longer there to get.
    sending_ = false;
    chat_over_ = true;
    return grpc::Status(grpc::StatusCode::CANCELLED, "The Send was cancelled.");
  }
  if (!turn.ok()) {
    // A request refused as invalid, or as a turn the context has no room for,
    // changed nothing. Anything else, a cancellation or a failure, may have
    // left the model with a turn the server never got: generating on would
    // build on what it cannot see.
    sending_ = false;
    if (absl::IsResourceExhausted(turn.status())) {
      AddNoRoomMetadata(turn.status(), *context);
    } else if (!absl::IsInvalidArgument(turn.status())) {
      chat_over_ = true;
    }
    return ToGrpc(turn.status());
  }
  capture_dir_ = turn->capture_dir;
  const std::filesystem::path turn_dir =
      data_dir_ / absl::StrCat("turn-", turn_number);
  if (absl::Status filed =
          FileTurn(turn->capture_dir, before, lines_before, turn_dir);
      !filed.ok()) {
    sending_ = false;
    chat_over_ = true;
    return ToGrpc(filed);
  }
  // Counted before the completion goes out: a client that has it may read
  // the capture at once. Until the write below returns, the turn shows in
  // GetState, and a ReadCaptures of it, which only a client guessing its
  // number makes, may succeed, though a cancel then takes the turn back.
  turns_ = turn_number;
  lock.unlock();
  // Without the lock: a write waits for the client to take it. A cancel that
  // came while the turn was filed leaves the client without the completion.
  protocol::SendResponse response;
  *response.mutable_completion() = std::move(turn->completion);
  const bool delivered = !context->IsCancelled() && writer->Write(response);
  lock.lock();
  sending_ = false;
  if (!delivered) {
    // As a cancel before the model finished: the model holds a turn the
    // server never got, which is no turn and has no capture.
    chat_over_ = true;
    turns_ = turn_number - 1;
    std::error_code error;
    std::filesystem::remove_all(turn_dir, error);
    return grpc::Status(grpc::StatusCode::CANCELLED, "The Send was cancelled.");
  }
  return grpc::Status::OK;
}

grpc::Status RunnerService::ReadCaptures(
    grpc::ServerContext* context, const protocol::ReadCapturesRequest* request,
    grpc::ServerWriter<protocol::CaptureChunk>* writer) {
  absl::StatusOr<std::filesystem::path> turn_dir;
  int chats = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    NoteClient(*context);
    turn_dir = TurnDirectory(request->turn());
    chats = chats_;
  }
  if (!turn_dir.ok()) return ToGrpc(turn_dir.status());
  // A turn released, or whose chat was closed, meanwhile has no capture,
  // whatever was sent of it.
  const auto finish = [&](grpc::Status status) {
    std::lock_guard<std::mutex> lock(mutex_);
    absl::StatusOr<std::filesystem::path> still =
        TurnDirectory(request->turn());
    if (still.ok() && chats_ != chats) {
      still = absl::NotFoundError(
          absl::StrCat("Turn ", request->turn(), " has no capture."));
    }
    if (!still.ok()) return ToGrpc(still.status());
    return status;
  };
  // Every entry is checked before the first file is sent.
  const absl::StatusOr<std::set<std::string>> names = CaptureFiles(*turn_dir);
  if (!names.ok()) return finish(ToGrpc(names.status()));
  const grpc::Status cancelled(grpc::StatusCode::CANCELLED,
                               "The read was cancelled.");
  // Read without the lock: a capture can be hundreds of megabytes, and the
  // window asks for the state meanwhile.
  std::string buffer(kCaptureChunkBytes, '\0');
  for (const std::string& name : *names) {
    const std::filesystem::path file = *turn_dir / name;
    std::ifstream in(file, std::ios::binary);
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(file, error);
    if (!in || error) {
      return finish(grpc::Status(grpc::StatusCode::INTERNAL,
                                 absl::StrCat("Cannot read ", file.string())));
    }
    protocol::CaptureChunk chunk;
    chunk.mutable_file()->set_path(name);
    chunk.mutable_file()->set_size(static_cast<int64_t>(size));
    if (!writer->Write(chunk)) return cancelled;
    Sha256 sha256;
    while (in.read(buffer.data(), kCaptureChunkBytes) || in.gcount() > 0) {
      const size_t count = static_cast<size_t>(in.gcount());
      sha256.Update(std::string_view(buffer.data(), count));
      chunk.set_data(buffer.data(), count);
      if (!writer->Write(chunk)) return cancelled;
    }
    if (in.bad()) {
      return finish(grpc::Status(grpc::StatusCode::INTERNAL,
                                 absl::StrCat("Cannot read ", file.string())));
    }
    chunk.set_sha256(sha256.Finish());
    if (!writer->Write(chunk)) return cancelled;
  }
  return finish(grpc::Status::OK);
}

grpc::Status RunnerService::ReleaseCaptures(
    grpc::ServerContext* context,
    const protocol::ReleaseCapturesRequest* request,
    protocol::ReleaseCapturesResponse* /*response*/) {
  std::lock_guard<std::mutex> lock(mutex_);
  NoteClient(*context);
  absl::StatusOr<std::filesystem::path> turn_dir =
      TurnDirectory(request->turn());
  if (!turn_dir.ok()) return ToGrpc(turn_dir.status());
  std::error_code error;
  std::filesystem::remove_all(*turn_dir, error);
  if (error) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        absl::StrCat("Cannot delete turn ", request->turn(),
                                     ": ", error.message()));
  }
  return grpc::Status::OK;
}

absl::StatusOr<std::filesystem::path> RunnerService::TurnDirectory(
    int turn) const {
  // Called with mutex_ held.
  const std::filesystem::path dir = data_dir_ / absl::StrCat("turn-", turn);
  std::error_code error;
  if (turn < 1 || turn > turns_ || !std::filesystem::is_directory(dir, error)) {
    return absl::NotFoundError(absl::StrCat("Turn ", turn, " has no capture."));
  }
  return dir;
}

State RunnerService::GetState() {
  std::lock_guard<std::mutex> lock(mutex_);
  State state;
  state.host = host_;
  state.client = client_;
  state.model_sha256 = model_sha256_;
  state.turns = turns_;
  state.chat_over = chat_over_;
  return state;
}

}  // namespace model_debugger::runner::service
// NOLINTEND
