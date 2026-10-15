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

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "grpcpp/grpcpp.h"
#include "model_debugger/runner/llm/runtime.h"
#include "model_debugger/runner/service/cache_directory.h"
#include "model_debugger/runner/service/host_info.h"
#include "model_debugger/v1/runner.grpc.pb.h"
#include "model_debugger/v1/runner.pb.h"
#include "openssl/sha.h"

namespace model_debugger::runner::service {
namespace {

// The version of the protocol this Runner was built against: the header of
// model_debugger/v1/runner.proto names it.
constexpr std::string_view kProtocolVersion = "1.0.0";

// The file the runtime appends a line to per decode step; the turn's lines are
// copied to the turn's directory.
constexpr std::string_view kTokensFile = "generated_tokens.jsonl";

// How many bytes of a captured file one CaptureData carries.
constexpr std::streamsize kCaptureDataBytes = 1 << 20;

// The maximum size of a data message in an UploadModel stream (1 MiB).
constexpr size_t kMaxUploadDataBytes = 1 << 20;

// The extension a model file is stored with: what the one runtime, LiteRT-LM,
// knows the format by.
constexpr std::string_view kModelExtension = ".litertlm";

// How often a Send checks whether its client has cancelled.
constexpr absl::Duration kCancelPollInterval = absl::Milliseconds(20);

// The refusal of a Load or CreateChat while another is in progress.
constexpr std::string_view kPreparing = "A Load or CreateChat is in progress.";

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

// Returns where `store` keeps the model of `sha256`: the one name UploadModel
// writes and Load reads.
std::filesystem::path ModelPath(const std::filesystem::path& store,
                                std::string_view sha256) {
  return store / absl::StrCat(sha256, kModelExtension);
}

// Returns the names of the files directly in `dir`.
std::set<std::string> FileNames(const std::filesystem::path& dir) {
  std::set<std::string> names;
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator(dir, error)) {
    if (entry.is_regular_file(error)) names.insert(entry.path().filename());
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

// Returns the SHA-256 of `file`, lower-case hexadecimal.
absl::StatusOr<std::string> Sha256Of(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) {
    return absl::InternalError(absl::StrCat("Cannot read ", file.string()));
  }
  Sha256 sha256;
  char buffer[1 << 16];
  while (in.read(buffer, sizeof(buffer)) || in.gcount() > 0) {
    sha256.Update(std::string_view(buffer, static_cast<size_t>(in.gcount())));
  }
  if (in.bad()) {
    return absl::InternalError(absl::StrCat("Cannot read ", file.string()));
  }
  return sha256.Finish();
}

// Moves what the turn wrote into its own directory: the files of
// `capture_dir` that were not there before, and the lines the tokens file
// gained. The runtime closes each tensor file after writing it and appends
// to the tokens file per line, so nothing is moved from under it.
absl::Status FileTurn(const std::filesystem::path& capture_dir,
                      const std::set<std::string>& before, int lines_before,
                      const std::filesystem::path& turn_dir) {
  std::error_code error;
  std::filesystem::create_directories(turn_dir, error);
  if (error) {
    return absl::InternalError(absl::StrCat("Cannot create ", turn_dir.string(),
                                            ": ", error.message()));
  }
  for (const std::string& name : FileNames(capture_dir)) {
    if (name == kTokensFile || before.count(name) != 0) continue;
    std::filesystem::rename(capture_dir / name, turn_dir / name, error);
    if (error) {
      return absl::InternalError(
          absl::StrCat("Cannot move ", name, ": ", error.message()));
    }
  }
  std::ifstream in(capture_dir / kTokensFile);
  std::ofstream out(turn_dir / kTokensFile);
  std::string line;
  for (int i = 0; std::getline(in, line); ++i) {
    if (i >= lines_before) out << line << '\n';
  }
  out.close();
  if (in.bad() || out.fail()) {
    return absl::InternalError(
        absl::StrCat("Cannot copy the turn's ", kTokensFile, "."));
  }
  return absl::OkStatus();
}

// Discards a partial upload.
void Discard(const std::filesystem::path& part) {
  std::error_code error;
  std::filesystem::remove(part, error);
}

// The status of an upload that could not be written, `reason` the errno:
// RESOURCE_EXHAUSTED when the disk is full, INTERNAL otherwise.
grpc::Status WriteFailure(int reason) {
  return grpc::Status(
      reason == ENOSPC ? grpc::StatusCode::RESOURCE_EXHAUSTED
                       : grpc::StatusCode::INTERNAL,
      absl::StrCat("Cannot write the model: ",
                   reason != 0 ? std::strerror(reason) : "unknown error"));
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
      absl::MutexLock lock(mutex_);
      stopped_ = true;
    }
    thread_.join();
  }

 private:
  // Runs on thread_ until the client cancels or the watcher is destroyed.
  void Watch(grpc::ServerContext& context, llm::Chat& chat) {
    absl::MutexLock lock(mutex_);
    while (!mutex_.AwaitWithTimeout(absl::Condition(&stopped_),
                                    kCancelPollInterval)) {
      if (context.IsCancelled()) {
        chat.Cancel();
        return;
      }
    }
  }

  absl::Mutex mutex_;
  bool stopped_ ABSL_GUARDED_BY(mutex_) = false;
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
    std::string_view pid_text =
        std::string_view(name).substr(kRunPrefix.size());
    pid_text = pid_text.substr(0, pid_text.find('.'));
    int pid = 0;
    if (!absl::SimpleAtoi(pid_text, &pid) || pid <= 0) {
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
  static std::atomic<int> instance{0};
  const int id = instance.fetch_add(1, std::memory_order_relaxed);
  const std::filesystem::path model_store = root / "models";
  const std::filesystem::path data_dir =
      root / (id == 0 ? absl::StrCat(kRunPrefix, getpid())
                      : absl::StrCat(kRunPrefix, getpid(), ".", id));
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
      host_(MeasureHost(runtime_->CanRunOnGpu())),
      model_store_(std::move(model_store)),
      data_dir_(std::move(data_dir)) {}

RunnerService::~RunnerService() {
  // The runtime, which writes there, goes first.
  chat_.reset();
  model_.reset();
  std::error_code error;
  std::filesystem::remove_all(data_dir_, error);
}

grpc::Status RunnerService::Admit(const grpc::ServerContext& context) {
  // Called with mutex_ held.
  if (client_.empty()) {
    client_ = context.peer();
  } else if (client_ != context.peer()) {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                        "This runner is serving another client.");
  }
  return grpc::Status::OK;
}

grpc::Status RunnerService::GetInfo(grpc::ServerContext* context,
                                    const v1::GetInfoRequest* /*request*/,
                                    v1::RunnerInfo* response) {
  absl::MutexLock lock(mutex_);
  if (grpc::Status admitted = Admit(*context); !admitted.ok()) return admitted;
  response->set_protocol_version(std::string(kProtocolVersion));
  *response->mutable_host() = host_;
  *response->mutable_runtime() = runtime_info_;
  return grpc::Status::OK;
}

grpc::Status RunnerService::UploadModel(
    grpc::ServerContext* context,
    grpc::ServerReader<v1::UploadModelRequest>* reader,
    v1::UploadModelResponse* /*response*/) {
  {
    absl::MutexLock lock(mutex_);
    if (grpc::Status admitted = Admit(*context); !admitted.ok()) {
      return admitted;
    }
  }
  v1::UploadModelRequest request;
  if (!reader->Read(&request) || !request.has_header()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "The stream does not start with a header.");
  }
  const v1::ModelHeader header = request.header();
  if (!IsSha256(header.sha256())) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "sha256 is not 64 lower-case hexadecimal digits.");
  }
  if (header.size() < 0) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "size is negative.");
  }
  // The store is shared by every upload and load; nothing in it changes
  // under a name once the name exists.
  std::error_code error;
  const std::filesystem::path file = ModelPath(model_store_, header.sha256());
  if (std::filesystem::is_regular_file(file, error)) return grpc::Status::OK;

  // Written under another name, kept only once the bytes prove to be the
  // model: a reader of the store never sees half a file.
  static std::atomic<int> upload_id{0};
  const std::filesystem::path part = std::filesystem::path(file).concat(
      absl::StrCat(".part.", getpid(), ".",
                   upload_id.fetch_add(1, std::memory_order_relaxed)));
  std::ofstream out(part, std::ios::binary | std::ios::trunc);
  if (!out) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        absl::StrCat("Cannot write ", part.string()));
  }
  Sha256 sha256;
  int64_t received = 0;
  while (reader->Read(&request)) {
    if (!request.has_data()) {
      Discard(part);
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "A second header arrived in the stream.");
    }
    if (request.data().size() > kMaxUploadDataBytes) {
      Discard(part);
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "A data message exceeds 1 MiB.");
    }
    const int64_t size = static_cast<int64_t>(request.data().size());
    if (size > header.size() - received) {
      Discard(part);
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          absl::StrCat("More than the ", header.size(),
                                       " bytes the header announced arrived."));
    }
    errno = 0;
    if (!out.write(request.data().data(),
                   static_cast<std::streamsize>(request.data().size()))) {
      const int reason = errno;
      Discard(part);
      return WriteFailure(reason);
    }
    sha256.Update(request.data());
    received += size;
  }
  // Closing flushes what the stream still holds: a full disk may show only
  // now.
  errno = 0;
  out.close();
  const int reason = errno;
  if (context->IsCancelled()) {
    Discard(part);
    return grpc::Status(grpc::StatusCode::CANCELLED,
                        "The upload was cancelled.");
  }
  if (out.fail()) {
    Discard(part);
    return WriteFailure(reason);
  }
  if (received != header.size()) {
    Discard(part);
    return grpc::Status(grpc::StatusCode::DATA_LOSS,
                        absl::StrCat("The header announced ", header.size(),
                                     " bytes; ", received, " arrived."));
  }
  if (const std::string digest = sha256.Finish(); digest != header.sha256()) {
    Discard(part);
    return grpc::Status(grpc::StatusCode::DATA_LOSS,
                        absl::StrCat("The bytes hash to ", digest, ", not to ",
                                     header.sha256(), "."));
  }
  std::filesystem::rename(part, file, error);
  if (error && !std::filesystem::is_regular_file(file)) {
    Discard(part);
    return grpc::Status(
        grpc::StatusCode::INTERNAL,
        absl::StrCat("Cannot keep the model: ", error.message()));
  }
  return grpc::Status::OK;
}

grpc::Status RunnerService::Load(grpc::ServerContext* context,
                                 const v1::LoadRequest* request,
                                 v1::LoadResponse* response) {
  std::filesystem::path model;
  {
    absl::MutexLock lock(mutex_);
    if (grpc::Status admitted = Admit(*context); !admitted.ok()) {
      return admitted;
    }
    if (model_ != nullptr) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          "A model is loaded; a runner loads one.");
    }
    if (loading_) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          std::string(kPreparing));
    }
    if (!IsSha256(request->sha256())) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "sha256 is not 64 lower-case hexadecimal digits.");
    }
    // NOT_FOUND asks the client to upload the model, so it means only that.
    model = ModelPath(model_store_, request->sha256());
    std::error_code error;
    const std::filesystem::file_status status =
        std::filesystem::status(model, error);
    if (status.type() == std::filesystem::file_type::not_found) {
      return grpc::Status(
          grpc::StatusCode::NOT_FOUND,
          absl::StrCat("No model with sha256 ", request->sha256(),
                       " in the model store."));
    }
    if (error) {
      return grpc::Status(grpc::StatusCode::INTERNAL,
                          absl::StrCat("Cannot look up ", model.string(), ": ",
                                       error.message()));
    }
    loading_ = true;
  }
  // Without the lock: loading takes seconds, and the window asks for the
  // state meanwhile.
  absl::StatusOr<std::unique_ptr<llm::Model>> loaded =
      runtime_->Load(model, data_dir_, *request);
  absl::MutexLock lock(mutex_);
  loading_ = false;
  if (context->IsCancelled()) {
    return grpc::Status(grpc::StatusCode::CANCELLED, "The Load was cancelled.");
  }
  if (absl::IsNotFound(loaded.status())) {
    // The model is in the store; what the runtime misses is the runner's.
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        std::string(loaded.status().message()));
  }
  if (!loaded.ok()) return ToGrpc(loaded.status());
  model_ = *std::move(loaded);
  model_sha256_ = request->sha256();
  *response = model_->Info();
  return grpc::Status::OK;
}

grpc::Status RunnerService::CreateChat(grpc::ServerContext* context,
                                       const v1::CreateChatRequest* request,
                                       v1::CreateChatResponse* /*response*/) {
  llm::Model* model = nullptr;
  {
    absl::MutexLock lock(mutex_);
    if (grpc::Status admitted = Admit(*context); !admitted.ok()) {
      return admitted;
    }
    if (loading_) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          std::string(kPreparing));
    }
    if (model_ == nullptr) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          "No model is loaded.");
    }
    if (chat_ != nullptr) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          "The chat exists; a runner holds one.");
    }
    loading_ = true;
    model = model_.get();
  }
  // Without the lock, like Load: the model is kept until the service goes.
  absl::StatusOr<std::unique_ptr<llm::Chat>> chat = model->CreateChat(*request);
  absl::MutexLock lock(mutex_);
  loading_ = false;
  if (context->IsCancelled()) {
    return grpc::Status(grpc::StatusCode::CANCELLED,
                        "The CreateChat was cancelled.");
  }
  if (!chat.ok()) return ToGrpc(chat.status());
  chat_ = *std::move(chat);
  return grpc::Status::OK;
}

grpc::Status RunnerService::Send(grpc::ServerContext* context,
                                 const v1::SendRequest* request,
                                 grpc::ServerWriter<v1::SendResponse>* writer) {
  llm::Chat* chat = nullptr;
  int turn_number = 0;
  std::filesystem::path capture_dir;
  {
    absl::MutexLock lock(mutex_);
    if (grpc::Status admitted = Admit(*context); !admitted.ok()) {
      return admitted;
    }
    if (chat_ == nullptr) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          "No chat exists.");
    }
    if (chat_over_) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          "The chat is over; generation on this connection "
                          "ended with a cancelled or failed turn.");
    }
    if (sending_) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          "A Send is in progress.");
    }
    sending_ = true;
    chat = chat_.get();
    turn_number = turns_ + 1;
    capture_dir = capture_dir_;
  }

  // What the capture directory holds before the turn, to tell the turn's
  // files apart afterwards. The directory is the chat's; the first turn
  // learns it from the result, so the snapshot is taken when known.
  std::set<std::string> before;
  int lines_before = 0;
  if (!capture_dir.empty()) {
    before = FileNames(capture_dir);
    lines_before = CountLines(capture_dir / kTokensFile);
  }

  absl::StatusOr<llm::Turn> turn;
  {
    CancelWatcher watcher(*context, *chat);
    turn = chat->Send(request->message(), [&](const v1::Chunk& chunk) {
      if (context->IsCancelled()) {
        chat->Cancel();
        return;
      }
      v1::SendResponse response;
      *response.mutable_chunk() = chunk;
      if (!writer->Write(response)) {
        chat->Cancel();
      }
    });
  }

  {
    absl::MutexLock lock(mutex_);
    sending_ = false;
    if (turn.ok() && context->IsCancelled()) {
      // The model finished a turn the client is no longer there to get.
      chat_over_ = true;
      return grpc::Status(grpc::StatusCode::CANCELLED,
                          "The Send was cancelled.");
    }
    if (!turn.ok()) {
      // A request refused as invalid changed nothing. Anything else, a
      // cancellation or a failure, may have left the model with a turn the
      // server never got: generating on would build on what it cannot see.
      if (!absl::IsInvalidArgument(turn.status())) chat_over_ = true;
      return ToGrpc(turn.status());
    }
    capture_dir_ = turn->capture_dir;
    const std::filesystem::path turn_dir =
        data_dir_ / absl::StrCat("turn-", turn_number);
    if (absl::Status filed =
            FileTurn(turn->capture_dir, before, lines_before, turn_dir);
        !filed.ok()) {
      chat_over_ = true;
      return ToGrpc(filed);
    }
    turns_ = turn_number;
  }
  v1::SendResponse response;
  *response.mutable_completion() = std::move(turn->completion);
  if (!writer->Write(response)) {
    absl::MutexLock lock(mutex_);
    chat_over_ = true;
    return grpc::Status(grpc::StatusCode::CANCELLED, "The Send was cancelled.");
  }
  return grpc::Status::OK;
}

grpc::Status RunnerService::ListCaptures(grpc::ServerContext* context,
                                         const v1::ListCapturesRequest* request,
                                         v1::ListCapturesResponse* response) {
  absl::StatusOr<std::filesystem::path> turn_dir;
  {
    absl::MutexLock lock(mutex_);
    if (grpc::Status admitted = Admit(*context); !admitted.ok()) {
      return admitted;
    }
    turn_dir = TurnDirectory(request->turn());
  }
  if (!turn_dir.ok()) return ToGrpc(turn_dir.status());
  // Hashed without the lock: a capture can be hundreds of megabytes, and the
  // window asks for the state meanwhile.
  absl::Status listed = absl::OkStatus();
  for (const std::string& name : FileNames(*turn_dir)) {
    const std::filesystem::path file = *turn_dir / name;
    absl::StatusOr<std::string> digest = Sha256Of(file);
    if (!digest.ok()) {
      listed = digest.status();
      break;
    }
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(file, error);
    if (error) {
      listed = absl::InternalError(absl::StrCat(
          "Cannot measure ", file.string(), ": ", error.message()));
      break;
    }
    v1::CaptureFile* entry = response->add_files();
    entry->set_path(name);
    entry->set_size(static_cast<int64_t>(size));
    entry->set_sha256(*std::move(digest));
  }
  // A turn released meanwhile has no capture, whatever was listed of it.
  absl::MutexLock lock(mutex_);
  if (absl::StatusOr<std::filesystem::path> still =
          TurnDirectory(request->turn());
      !still.ok()) {
    return ToGrpc(still.status());
  }
  return ToGrpc(listed);
}

grpc::Status RunnerService::ReadCapture(
    grpc::ServerContext* context, const v1::ReadCaptureRequest* request,
    grpc::ServerWriter<v1::CaptureData>* writer) {
  std::filesystem::path file;
  {
    absl::MutexLock lock(mutex_);
    if (grpc::Status admitted = Admit(*context); !admitted.ok()) {
      return admitted;
    }
    absl::StatusOr<std::filesystem::path> turn_dir =
        TurnDirectory(request->turn());
    if (!turn_dir.ok()) return ToGrpc(turn_dir.status());
    // A name, not a path: nothing outside the turn's directory is readable.
    if (request->path().empty() ||
        request->path() != std::filesystem::path(request->path()).filename()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "path is the name of a file in the turn's capture.");
    }
    file = *turn_dir / request->path();
  }
  std::error_code error;
  std::ifstream in(file, std::ios::binary);
  if (!std::filesystem::is_regular_file(file, error) || !in) {
    return grpc::Status(grpc::StatusCode::NOT_FOUND,
                        absl::StrCat("No file ", request->path(), " in turn ",
                                     request->turn(), "."));
  }
  std::string buffer(kCaptureDataBytes, '\0');
  while (in.read(buffer.data(), kCaptureDataBytes) || in.gcount() > 0) {
    v1::CaptureData data;
    data.set_data(buffer.data(), static_cast<size_t>(in.gcount()));
    if (!writer->Write(data)) {
      return grpc::Status(grpc::StatusCode::CANCELLED,
                          "The read was cancelled.");
    }
  }
  if (in.bad()) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        absl::StrCat("Cannot read ", file.string()));
  }
  return grpc::Status::OK;
}

grpc::Status RunnerService::ReleaseCaptures(
    grpc::ServerContext* context, const v1::ReleaseCapturesRequest* request,
    v1::ReleaseCapturesResponse* /*response*/) {
  absl::MutexLock lock(mutex_);
  if (grpc::Status admitted = Admit(*context); !admitted.ok()) return admitted;
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
  absl::MutexLock lock(mutex_);
  State state;
  state.host = host_;
  state.client = client_;
  state.model_sha256 = model_sha256_;
  state.turns = turns_;
  state.chat_over = chat_over_;
  return state;
}

}  // namespace model_debugger::runner::service
