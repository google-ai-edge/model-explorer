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

// The service's own rules, which hold whatever the runtime: on a runtime the
// test scripts, which fails, waits or finishes when the test says. Each test
// has a service of its own, since a failed Load is final.

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

// NOLINTBEGIN(build/c++11)
// NOLINTBEGIN(build/c++17)
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "google/protobuf/struct.pb.h"
#include "grpcpp/grpcpp.h"
#include "model_debugger/protocol/runner.grpc.pb.h"
#include "model_debugger/protocol/runner.pb.h"
#include "model_debugger/runner/llm/runtime.h"
#include "model_debugger/runner/service/model_store.h"
#include "model_debugger/runner/service/runner_service.h"
#include "model_debugger/runner/testing/test_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"

namespace model_debugger::runner::service {
namespace {

// A model the test's store holds, by its name.
constexpr std::string_view kSha256 =
    "9f3a7c1e5b2d4a6f8e0c1b3d5f7a9c2e4b6d8f0a1c3e5b7d9f2a4c6e8b0d1f3a";

// Five bytes and their SHA-256: a model small enough to upload in a test.
constexpr std::string_view kHello = "hello";
constexpr std::string_view kHelloSha256 =
    "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

// The name the store keeps the model of `sha256` by.
std::string ModelName(std::string_view sha256) {
  return absl::StrCat(sha256, ".litertlm");
}

// The service's refusals, as the client reads them.
constexpr std::string_view kPreparing = "A Load or CreateChat is in progress.";
constexpr std::string_view kLoadFailed =
    "A Load failed; a runner loads one model, once.";
constexpr std::string_view kChatOver =
    "The chat has ended with a cancelled or failed turn; a CreateChat starts "
    "another.";

// The schemas the fake runtime reports, which the service checks options
// against.
constexpr std::string_view kLoadSchema = R"({
  "type": "object",
  "properties": {"threads": {"type": "integer", "minimum": 1}},
  "additionalProperties": false
})";
constexpr std::string_view kChatSchema = R"({
  "type": "object",
  "properties": {"mode": {"enum": ["a", "b"]}},
  "additionalProperties": false
})";

// Where a fake call waits until the test lets it go on.
class Gate {
 public:
  // Notes that the call arrived, then waits for Open.
  void Pass() {
    std::unique_lock<std::mutex> lock(mutex_);
    arrived_ = true;
    changed_.notify_all();
    changed_.wait(lock, [this] { return open_; });
  }

  // Waits until a call has arrived.
  void AwaitArrival() {
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock, [this] { return arrived_; });
  }

  void Open() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      open_ = true;
    }
    changed_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable changed_;
  bool arrived_ = false;
  bool open_ = false;
};

// What the fake runtime does, set by the test before each call.
struct Script {
  // What CanRunOnGpu returns, which the service reads once it is created.
  absl::Status gpu = absl::FailedPreconditionError("The fake has no GPU.");
  absl::Status load;          // What Load returns; OK gives a model.
  absl::Status create_chat;   // What CreateChat returns; OK gives a chat.
  Gate* load_gate = nullptr;  // Where Load waits, if set.
  Gate* chat_gate = nullptr;  // Where CreateChat waits, if set.
  absl::Status send;          // What Send returns; OK gives a turn.
  Gate* send_gate = nullptr;  // Where Send waits, if set.
  std::filesystem::path capture_dir;  // Where every turn says it wrote.
  protocol::LoadResponse info;        // What the model loaded says of itself.
  // The chats not yet deleted, and how many there were when CreateChat was
  // last called.
  mutable std::atomic<int> chats{0};
  mutable std::atomic<int> chats_at_create{-1};
};

// A chat that replies "Hi" to every turn.
class FakeChat : public llm::Chat {
 public:
  explicit FakeChat(const Script& script) : script_(script) { ++script_.chats; }
  ~FakeChat() override { --script_.chats; }

  absl::StatusOr<llm::Turn> Send(const protocol::ChatMessage& /*message*/,
                                 const llm::ChunkCallback& on_chunk) override {
    if (script_.send_gate != nullptr) script_.send_gate->Pass();
    if (!script_.send.ok()) return script_.send;
    protocol::Chunk chunk;
    chunk.set_content("Hi");
    on_chunk(chunk);
    llm::Turn turn;
    turn.completion.set_content("Hi");
    turn.completion.mutable_model_debugger()->set_turn(++turns_);
    turn.capture_dir = script_.capture_dir;
    return turn;
  }

  // The reply is one chunk: nothing to stop.
  void Cancel() override {}

 private:
  const Script& script_;
  int turns_ = 0;
};

class FakeModel : public llm::Model {
 public:
  explicit FakeModel(const Script& script)
      : script_(script), info_(script.info) {}

  const protocol::LoadResponse& Info() const override { return info_; }

  absl::StatusOr<std::unique_ptr<llm::Chat>> CreateChat(
      const protocol::CreateChatRequest& /*request*/) override {
    script_.chats_at_create = script_.chats.load();
    if (script_.chat_gate != nullptr) script_.chat_gate->Pass();
    if (!script_.create_chat.ok()) return script_.create_chat;
    return std::make_unique<FakeChat>(script_);
  }

 private:
  const Script& script_;
  const protocol::LoadResponse info_;
};

class FakeRuntime : public llm::Runtime {
 public:
  explicit FakeRuntime(const Script& script) : script_(script) {
    info_.set_load_options_schema(std::string(kLoadSchema));
    info_.set_chat_options_schema(std::string(kChatSchema));
  }

  const protocol::RuntimeInfo& Info() const override { return info_; }

  absl::Status CanRunOnGpu() const override { return script_.gpu; }

  absl::StatusOr<std::unique_ptr<llm::Model>> Load(
      const std::filesystem::path& /*model*/,
      const std::filesystem::path& /*data_dir*/,
      const protocol::LoadRequest& /*request*/) override {
    if (script_.load_gate != nullptr) script_.load_gate->Pass();
    if (!script_.load.ok()) return script_.load;
    return std::make_unique<FakeModel>(script_);
  }

 private:
  const Script& script_;
  protocol::RuntimeInfo info_;
};

protocol::LoadRequest LoadOf(std::string_view sha256) {
  protocol::LoadRequest request;
  request.set_sha256(std::string(sha256));
  request.set_device(protocol::CPU);
  request.set_context_length(1024);
  return request;
}

protocol::CreateChatRequest ChatOf(const protocol::ChatOptions& options) {
  protocol::CreateChatRequest request;
  *request.mutable_options() = options;
  return request;
}

protocol::ChatOptions Options() {
  protocol::ChatOptions options;
  options.set_max_completion_tokens(8);
  return options;
}

google::protobuf::Value Number(double number) {
  google::protobuf::Value value;
  value.set_number_value(number);
  return value;
}

google::protobuf::Value Text(const char* text) {
  google::protobuf::Value value;
  value.set_string_value(text);
  return value;
}

// Notes the server side of each call as it starts, so that a test can wait
// until the service sees the call cancelled. Adds no interceptor.
class CallRecorder
    : public grpc::experimental::ServerInterceptorFactoryInterface {
 public:
  explicit CallRecorder(std::atomic<grpc::ServerContextBase*>& latest)
      : latest_(latest) {}

  grpc::experimental::Interceptor* CreateServerInterceptor(
      grpc::experimental::ServerRpcInfo* info) override {
    latest_.store(info->server_context());
    return nullptr;
  }

 private:
  std::atomic<grpc::ServerContextBase*>& latest_;
};

class RunnerServiceRulesTest : public testing::Test {
 protected:
  void SetUp() override {
    const std::filesystem::path cache = UseTestCacheDirectory();
    store_ = cache / "models";
    std::filesystem::create_directories(store_);
    std::ofstream(store_ / ModelName(kSha256));
    run_dir_ = cache / absl::StrCat("run.", getpid());
    script_.capture_dir = TempDir("capture");
    absl::StatusOr<std::unique_ptr<RunnerService>> service =
        RunnerService::Create(std::make_unique<FakeRuntime>(script_));
    ASSERT_TRUE(service.ok()) << service.status();
    service_ = *std::move(service);
    grpc::ServerBuilder builder;
    std::vector<
        std::unique_ptr<grpc::experimental::ServerInterceptorFactoryInterface>>
        recorders;
    recorders.push_back(std::make_unique<CallRecorder>(latest_call_));
    builder.experimental().SetInterceptorCreators(std::move(recorders));
    int port = 0;
    builder.AddListeningPort(
        "127.0.0.1:0", grpc::InsecureServerCredentials(),
        &port);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    ASSERT_NE(server_, nullptr);
    stub_ = protocol::Runner::NewStub(
        grpc::CreateChannel("127.0.0.1:" + std::to_string(port),
                            grpc::InsecureChannelCredentials()));
  }

  void TearDown() override {
    if (server_ != nullptr) server_->Shutdown();
  }

  grpc::Status Load(const protocol::LoadRequest& request = LoadOf(kSha256)) {
    grpc::ClientContext context;
    protocol::LoadResponse response;
    return stub_->Load(&context, request, &response);
  }

  // Uploads kHello, in one piece.
  grpc::Status UploadHello() {
    grpc::ClientContext context;
    protocol::UploadModelResponse response;
    std::unique_ptr<grpc::ClientWriter<protocol::UploadModelRequest>> writer =
        stub_->UploadModel(&context, &response);
    protocol::UploadModelRequest request;
    request.mutable_header()->set_sha256(std::string(kHelloSha256));
    request.mutable_header()->set_size(kHello.size());
    writer->Write(request);
    request.set_data(std::string(kHello));
    writer->Write(request);
    writer->WritesDone();
    return writer->Finish();
  }

  // The names of the files in the store.
  std::set<std::string> StoreNames() const {
    std::set<std::string> names;
    for (const auto& entry : std::filesystem::directory_iterator(store_)) {
      names.insert(entry.path().filename());
    }
    return names;
  }

  // Creates a chat; `metadata`, if set, gets the call's trailing metadata.
  grpc::Status CreateChat(
      const protocol::ChatOptions& options = Options(),
      std::multimap<std::string, std::string>* metadata = nullptr) {
    grpc::ClientContext context;
    protocol::CreateChatResponse response;
    const grpc::Status status =
        stub_->CreateChat(&context, ChatOf(options), &response);
    if (metadata != nullptr) TrailingMetadata(context, *metadata);
    return status;
  }

  // Sends a turn; `metadata`, if set, gets the call's trailing metadata.
  grpc::Status Send(
      std::multimap<std::string, std::string>* metadata = nullptr) {
    grpc::ClientContext context;
    protocol::SendRequest request;
    request.mutable_message()->set_role(protocol::ChatMessage::USER);
    request.mutable_message()->set_content("Hello?");
    std::unique_ptr<grpc::ClientReader<protocol::SendResponse>> reader =
        stub_->Send(&context, request);
    protocol::SendResponse response;
    while (reader->Read(&response)) {
    }
    const grpc::Status status = reader->Finish();
    if (metadata != nullptr) TrailingMetadata(context, *metadata);
    return status;
  }

  // Gives `metadata` the trailing metadata of the call `context` made.
  static void TrailingMetadata(
      const grpc::ClientContext& context,
      std::multimap<std::string, std::string>& metadata) {
    for (const auto& [key, value] : context.GetServerTrailingMetadata()) {
      metadata.emplace(std::string(key.data(), key.size()),
                       std::string(value.data(), value.size()));
    }
  }

  // Reads turn `turn`'s capture; `files` counts the files the stream began.
  grpc::Status ReadCaptures(int turn, int* files = nullptr) {
    grpc::ClientContext context;
    protocol::ReadCapturesRequest request;
    request.set_turn(turn);
    std::unique_ptr<grpc::ClientReader<protocol::CaptureChunk>> reader =
        stub_->ReadCaptures(&context, request);
    protocol::CaptureChunk chunk;
    int started = 0;
    while (reader->Read(&chunk)) {
      if (chunk.has_file()) ++started;
    }
    if (files != nullptr) *files = started;
    return reader->Finish();
  }

  // Waits until the service sees its latest call cancelled. Only while the
  // service holds that call, which keeps its context alive.
  bool AwaitCancelSeen() {
    grpc::ServerContextBase* call = latest_call_.load();
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!call->IsCancelled() &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return call->IsCancelled();
  }

  // Calls `call` until it is not refused as a Load or CreateChat in
  // progress: one the test let go on is finishing.
  static grpc::Status Once(const std::function<grpc::Status()>& call) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    grpc::Status status = call();
    while (status.error_message() == kPreparing &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      status = call();
    }
    return status;
  }

  State state() { return service_->GetState(); }

  Script script_;  // First: the runtime reads it until the service goes.
  std::filesystem::path store_;    // The model store.
  std::filesystem::path run_dir_;  // Where the service keeps each turn.
  std::atomic<grpc::ServerContextBase*> latest_call_{nullptr};
  std::unique_ptr<RunnerService> service_;
  std::unique_ptr<protocol::Runner::Stub> stub_;
  std::unique_ptr<grpc::Server> server_;  // Last: it goes first.
};

TEST_F(RunnerServiceRulesTest, ALoadOfAModelNotInTheStoreIsNoFailure) {
  EXPECT_EQ(Load(LoadOf(std::string(64, '0'))).error_code(),
            grpc::StatusCode::NOT_FOUND);
  EXPECT_THAT(StoreNames(), testing::IsEmpty())
      << "the Load named another model";
  std::ofstream(store_ / ModelName(kSha256));  // The upload.
  EXPECT_TRUE(Load().ok()) << "the Load again";
  const grpc::Status again = Load();
  EXPECT_EQ(again.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  EXPECT_EQ(again.error_message(), "A model is loaded; a runner loads one.");
}

TEST_F(RunnerServiceRulesTest, ALoadKeepsOnlyItsModelInTheStore) {
  std::ofstream(store_ / ModelName(kHelloSha256));
  std::ofstream(store_ / (ModelName(kHelloSha256) + ".part"));  // A crash's.
  ASSERT_TRUE(Load().ok());
  EXPECT_THAT(StoreNames(), testing::ElementsAre(ModelName(kSha256)));
}

TEST_F(RunnerServiceRulesTest, AnUploadedModelIsHeldUntilAnotherIsNamed) {
  ASSERT_TRUE(UploadHello().ok());
  EXPECT_THAT(StoreNames(), testing::ElementsAre(ModelName(kHelloSha256)))
      << "the upload named its model";
  std::ofstream(store_ / ModelName(kSha256));
  // Another Runner sharing the store names another model meanwhile.
  ModelStore(store_).KeepOnly(kSha256);
  EXPECT_THAT(StoreNames(), testing::ElementsAre(ModelName(kHelloSha256),
                                                 ModelName(kSha256)));
  ASSERT_TRUE(Load().ok());
  EXPECT_THAT(StoreNames(), testing::ElementsAre(ModelName(kSha256)))
      << "this runner named another model";
}

TEST_F(RunnerServiceRulesTest, TheLoadedModelIsHeldWhenAnotherIsNamed) {
  ASSERT_TRUE(Load().ok());
  ASSERT_TRUE(UploadHello().ok());
  // And when another Runner sharing the store names a third.
  ModelStore(store_).KeepOnly(std::string(64, '0'));
  EXPECT_THAT(StoreNames(), testing::ElementsAre(ModelName(kHelloSha256),
                                                 ModelName(kSha256)));
}

TEST_F(RunnerServiceRulesTest, ALoadAnswersWhatTheModelSaysOfItself) {
  script_.info.set_start_token(2);
  protocol::SamplerDefaults& defaults =
      *script_.info.mutable_sampler_defaults();
  defaults.set_type(protocol::SamplerDefaults::TOP_K);
  defaults.set_top_k(40);
  grpc::ClientContext context;
  protocol::LoadResponse response;
  ASSERT_TRUE(stub_->Load(&context, LoadOf(kSha256), &response).ok());
  EXPECT_EQ(response.SerializeAsString(), script_.info.SerializeAsString());
}

TEST_F(RunnerServiceRulesTest, AFailedLoadIsFinal) {
  script_.load = absl::InternalError("The engine broke.");
  EXPECT_EQ(Load().error_code(), grpc::StatusCode::INTERNAL);
  script_.load = absl::OkStatus();
  const grpc::Status again = Load();
  EXPECT_EQ(again.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  EXPECT_EQ(again.error_message(), kLoadFailed);
  EXPECT_TRUE(state().model_sha256.empty());
}

TEST_F(RunnerServiceRulesTest, ALoadWithOptionsTheSchemaRefusesIsFinal) {
  protocol::LoadRequest request = LoadOf(kSha256);
  (*request.mutable_runtime_options()->mutable_fields())["threads"] = Number(0);
  const grpc::Status refused = Load(request);
  EXPECT_EQ(refused.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(refused.error_message(),
            "LoadRequest.runtime_options.threads is 0.0; it must be at least "
            "1.");
  EXPECT_EQ(Load().error_message(), kLoadFailed);
}

TEST_F(RunnerServiceRulesTest, OptionsAreCheckedBeforeTheStore) {
  protocol::LoadRequest request = LoadOf(std::string(64, '0'));
  (*request.mutable_runtime_options()->mutable_fields())["turbo"] = Number(1);
  EXPECT_EQ(Load(request).error_code(), grpc::StatusCode::INVALID_ARGUMENT)
      << "no upload for a Load that would be refused";
}

TEST_F(RunnerServiceRulesTest, AContextLengthBelowOneIsRefusedBeforeTheStore) {
  protocol::LoadRequest request = LoadOf(std::string(64, '0'));
  request.set_context_length(0);
  const grpc::Status refused = Load(request);
  EXPECT_EQ(refused.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(refused.error_message(),
            "LoadRequest.context_length is 0; it must be positive.");
}

TEST_F(RunnerServiceRulesTest, ADeviceNotNamedIsRefusedBeforeTheStore) {
  protocol::LoadRequest request = LoadOf(std::string(64, '0'));
  request.set_device(protocol::DEVICE_UNSPECIFIED);
  const grpc::Status refused = Load(request);
  EXPECT_EQ(refused.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(refused.error_message(), "LoadRequest.device is not CPU or GPU.");
}

TEST_F(RunnerServiceRulesTest, AGpuTheRuntimeCannotUseIsNoneAndRefused) {
  grpc::ClientContext context;
  protocol::RunnerInfo info;
  ASSERT_TRUE(stub_->GetInfo(&context, protocol::GetInfoRequest(), &info).ok());
  EXPECT_FALSE(info.host().has_gpu());
  protocol::LoadRequest request = LoadOf(std::string(64, '0'));
  request.set_device(protocol::GPU);
  const grpc::Status refused = Load(request);
  EXPECT_EQ(refused.error_code(), grpc::StatusCode::FAILED_PRECONDITION)
      << "before the store, which lacks the model";
  EXPECT_EQ(refused.error_message(), "The fake has no GPU.");
}

TEST_F(RunnerServiceRulesTest, ALoadPastItsDeadlineLoadsNoModel) {
  Gate gate;
  script_.load_gate = &gate;
  grpc::Status late;
  std::thread call([&] {
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() +
                         std::chrono::milliseconds(500));
    protocol::LoadResponse response;
    late = stub_->Load(&context, LoadOf(kSha256), &response);
  });
  gate.AwaitArrival();
  call.join();
  EXPECT_EQ(late.error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
  ASSERT_TRUE(AwaitCancelSeen());
  gate.Open();  // The runtime loads the model all the same.
  const grpc::Status again = Once([this] { return Load(); });
  EXPECT_EQ(again.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  EXPECT_EQ(again.error_message(), kLoadFailed);
  EXPECT_TRUE(state().model_sha256.empty()) << "the model was discarded";
}

TEST_F(RunnerServiceRulesTest, AFailedCreateChatIsNotFinal) {
  ASSERT_TRUE(Load().ok());
  script_.create_chat = absl::InternalError("The session broke.");
  EXPECT_EQ(CreateChat().error_code(), grpc::StatusCode::INTERNAL);
  EXPECT_EQ(Send().error_message(), "No chat exists.");
  script_.create_chat = absl::OkStatus();
  const grpc::Status again = CreateChat();
  EXPECT_TRUE(again.ok()) << again.error_message();
  EXPECT_TRUE(Send().ok());
}

TEST_F(RunnerServiceRulesTest, ACancelledCreateChatCreatesNoChat) {
  ASSERT_TRUE(Load().ok());
  Gate gate;
  script_.chat_gate = &gate;
  grpc::ClientContext context;
  grpc::Status cancelled;
  std::thread call([&] {
    protocol::CreateChatResponse response;
    cancelled = stub_->CreateChat(&context, ChatOf(Options()), &response);
  });
  gate.AwaitArrival();
  context.TryCancel();
  call.join();
  EXPECT_EQ(cancelled.error_code(), grpc::StatusCode::CANCELLED);
  ASSERT_TRUE(AwaitCancelSeen());
  gate.Open();  // The runtime creates the chat all the same.
  script_.chat_gate = nullptr;
  const grpc::Status again = Once([this] { return CreateChat(); });
  EXPECT_TRUE(again.ok()) << again.error_message();
  EXPECT_EQ(script_.chats_at_create, 0) << "the cancelled chat was discarded";
}

TEST_F(RunnerServiceRulesTest, ChatOptionsAtTheirBoundsAreTaken) {
  ASSERT_TRUE(Load().ok());
  protocol::ChatOptions options;
  options.set_max_completion_tokens(1);
  options.set_temperature(0);
  options.set_top_p(1);
  options.set_top_k(1);
  options.set_seed(-7);
  const grpc::Status status = CreateChat(options);
  EXPECT_TRUE(status.ok()) << status.error_message();
}

TEST_F(RunnerServiceRulesTest, ChatRuntimeOptionsAreCheckedAgainstTheSchema) {
  ASSERT_TRUE(Load().ok());
  protocol::ChatOptions options = Options();
  (*options.mutable_runtime_options()->mutable_fields())["mode"] = Text("c");
  const grpc::Status refused = CreateChat(options);
  EXPECT_EQ(refused.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(refused.error_message(),
            "ChatOptions.runtime_options.mode is \"c\"; it must be one of "
            "[\"a\",\"b\"].");
  EXPECT_TRUE(CreateChat().ok()) << "a refusal is not final";
}

TEST_F(RunnerServiceRulesTest, AFlatCaptureIsReadWhole) {
  ASSERT_TRUE(Load().ok());
  ASSERT_TRUE(CreateChat().ok());
  std::ofstream(script_.capture_dir / "tensor.safetensors") << "bytes";
  std::ofstream(script_.capture_dir / "generated_tokens.jsonl") << "{}\n";
  ASSERT_TRUE(Send().ok());
  int files = 0;
  const grpc::Status read = ReadCaptures(1, &files);
  EXPECT_TRUE(read.ok()) << read.error_message();
  EXPECT_EQ(files, 2);
}

TEST_F(RunnerServiceRulesTest, ACaptureWithADirectoryInItIsNotReadInPart) {
  ASSERT_TRUE(Load().ok());
  ASSERT_TRUE(CreateChat().ok());
  std::ofstream(script_.capture_dir / "tensor.safetensors") << "bytes";
  std::filesystem::create_directories(script_.capture_dir / "nested");
  std::ofstream(script_.capture_dir / "nested" / "inner.safetensors")
      << "bytes";
  ASSERT_TRUE(Send().ok());
  int files = -1;
  const grpc::Status read = ReadCaptures(1, &files);
  EXPECT_EQ(read.error_code(), grpc::StatusCode::INTERNAL);
  EXPECT_EQ(read.error_message(),
            (run_dir_ / "turn-1" / "nested").string() +
                " is not a regular file; a capture is the files directly in "
                "the turn's directory.");
  EXPECT_EQ(files, 0) << "no file of it is sent";
}

TEST_F(RunnerServiceRulesTest, ATurnWithNoRoomIsRefusedAndTheChatGoesOn) {
  ASSERT_TRUE(Load().ok());
  ASSERT_TRUE(CreateChat().ok());
  absl::Status no_room = absl::ResourceExhaustedError(
      "This turn needs 900 tokens; the context has 700 left.");
  no_room.SetPayload(llm::kTurnTokensPayload, absl::Cord("900"));
  no_room.SetPayload(llm::kTokensLeftPayload, absl::Cord("700"));
  script_.send = no_room;
  std::multimap<std::string, std::string> metadata;
  const grpc::Status refused = Send(&metadata);
  EXPECT_EQ(refused.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
  EXPECT_EQ(refused.error_message(),
            "This turn needs 900 tokens; the context has 700 left.");
  EXPECT_THAT(metadata, testing::IsSupersetOf({
                            testing::Pair("turn-tokens", "900"),
                            testing::Pair("tokens-left", "700"),
                        }));
  EXPECT_FALSE(state().chat_over);
  EXPECT_EQ(state().turns, 0) << "a refused turn is no turn";

  script_.send = absl::OkStatus();
  const grpc::Status next = Send();
  EXPECT_TRUE(next.ok()) << next.error_message();
  EXPECT_EQ(state().turns, 1);
}

TEST_F(RunnerServiceRulesTest, AChatWithNoRoomToStartIsRefusedWithItsCounts) {
  ASSERT_TRUE(Load().ok());
  ASSERT_TRUE(CreateChat().ok());
  absl::Status no_room = absl::ResourceExhaustedError(
      "The starting messages need 5000 tokens in the chat template; the "
      "context has 1024.");
  no_room.SetPayload(llm::kTurnTokensPayload, absl::Cord("5000"));
  no_room.SetPayload(llm::kTokensLeftPayload, absl::Cord("1024"));
  script_.create_chat = no_room;
  std::multimap<std::string, std::string> metadata;
  const grpc::Status refused = CreateChat(Options(), &metadata);
  EXPECT_EQ(refused.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
  EXPECT_THAT(metadata, testing::IsSupersetOf({
                            testing::Pair("turn-tokens", "5000"),
                            testing::Pair("tokens-left", "1024"),
                        }));
  EXPECT_EQ(script_.chats, 0) << "the old chat is closed all the same";
  EXPECT_EQ(Send().error_message(), "No chat exists.");

  script_.create_chat = absl::OkStatus();
  EXPECT_TRUE(CreateChat().ok()) << "a refusal is not final";
}

TEST_F(RunnerServiceRulesTest, AFailedTurnEndsTheChat) {
  ASSERT_TRUE(Load().ok());
  ASSERT_TRUE(CreateChat().ok());
  script_.send = absl::InternalError("The runtime failed.");
  EXPECT_EQ(Send().error_code(), grpc::StatusCode::INTERNAL);
  EXPECT_TRUE(state().chat_over);
  script_.send = absl::OkStatus();
  const grpc::Status again = Send();
  EXPECT_EQ(again.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  EXPECT_EQ(again.error_message(), kChatOver);
}

TEST_F(RunnerServiceRulesTest, ACancelAfterTheModelFinishedEndsTheChat) {
  ASSERT_TRUE(Load().ok());
  ASSERT_TRUE(CreateChat().ok());
  // The turn's tokens file as a pipe: the service, filing the turn after the
  // model finished, waits on it until the test closes it.
  const std::filesystem::path tokens =
      script_.capture_dir / "generated_tokens.jsonl";
  ASSERT_EQ(mkfifo(tokens.c_str(), 0600), 0);
  std::ofstream(script_.capture_dir / "tensor.safetensors") << "bytes";
  grpc::ClientContext context;
  grpc::Status cancelled;
  std::thread send([&] {
    protocol::SendRequest request;
    request.mutable_message()->set_role(protocol::ChatMessage::USER);
    request.mutable_message()->set_content("Hello?");
    std::unique_ptr<grpc::ClientReader<protocol::SendResponse>> reader =
        stub_->Send(&context, request);
    protocol::SendResponse response;
    while (reader->Read(&response)) {
    }
    cancelled = reader->Finish();
  });
  // Opens once the service reads the pipe: the model has finished.
  const int pipe = open(tokens.c_str(), O_WRONLY);
  ASSERT_GE(pipe, 0);
  context.TryCancel();
  send.join();
  EXPECT_EQ(cancelled.error_code(), grpc::StatusCode::CANCELLED);
  ASSERT_TRUE(AwaitCancelSeen());
  close(pipe);  // The service files the turn and finds the call cancelled.

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!state().chat_over && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(state().chat_over);
  EXPECT_EQ(state().turns, 0) << "a cancelled turn is no turn";
  EXPECT_EQ(ReadCaptures(1).error_code(), grpc::StatusCode::NOT_FOUND)
      << "a cancelled turn has no capture";
  EXPECT_FALSE(std::filesystem::exists(run_dir_ / "turn-1"))
      << "nor any files left";
  const grpc::Status again = Send();
  EXPECT_EQ(again.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  EXPECT_EQ(again.error_message(), kChatOver);
}

TEST_F(RunnerServiceRulesTest, ACreateChatReplacesTheChat) {
  ASSERT_TRUE(Load().ok());
  ASSERT_TRUE(CreateChat().ok());
  const std::filesystem::path first_dir = script_.capture_dir;
  for (int turn = 1; turn <= 2; ++turn) {
    std::ofstream(first_dir / "tensor.safetensors") << "bytes";
    std::ofstream(first_dir / "generated_tokens.jsonl", std::ios::app)
        << "{\"old\":" << turn << "}\n";
    ASSERT_TRUE(Send().ok());
  }
  script_.capture_dir = TempDir("capture_2");  // The next chat's.
  const grpc::Status replaced = CreateChat();
  ASSERT_TRUE(replaced.ok()) << replaced.error_message();
  EXPECT_EQ(script_.chats_at_create, 0)
      << "the old chat is deleted before the new one is created";
  EXPECT_EQ(script_.chats, 1);
  EXPECT_EQ(state().turns, 0);
  for (int turn = 1; turn <= 2; ++turn) {
    EXPECT_EQ(ReadCaptures(turn).error_code(), grpc::StatusCode::NOT_FOUND)
        << "turn " << turn << " of the old chat";
    EXPECT_FALSE(
        std::filesystem::exists(run_dir_ / absl::StrCat("turn-", turn)));
  }
  EXPECT_FALSE(std::filesystem::exists(first_dir))
      << "nor is anything left in its capture directory";

  std::ofstream(script_.capture_dir / "tensor.safetensors") << "bytes";
  std::ofstream(script_.capture_dir / "generated_tokens.jsonl")
      << "{\"new\":1}\n";
  ASSERT_TRUE(Send().ok());
  EXPECT_EQ(state().turns, 1) << "the new chat's turns count from 1";
  int files = 0;
  const grpc::Status read = ReadCaptures(1, &files);
  EXPECT_TRUE(read.ok()) << read.error_message();
  EXPECT_EQ(files, 2) << "the new chat's files, the same names as the old's";
  std::ifstream tokens(run_dir_ / "turn-1" / "generated_tokens.jsonl");
  EXPECT_EQ(std::string(std::istreambuf_iterator<char>(tokens), {}),
            "{\"new\":1}\n")
      << "the new chat's lines, from its first";
}

TEST_F(RunnerServiceRulesTest, ACreateChatReplacesAChatThatIsOver) {
  ASSERT_TRUE(Load().ok());
  ASSERT_TRUE(CreateChat().ok());
  script_.send = absl::InternalError("The runtime failed.");
  EXPECT_EQ(Send().error_code(), grpc::StatusCode::INTERNAL);
  ASSERT_TRUE(state().chat_over);
  script_.send = absl::OkStatus();
  const grpc::Status replaced = CreateChat();
  ASSERT_TRUE(replaced.ok()) << replaced.error_message();
  EXPECT_FALSE(state().chat_over);
  EXPECT_TRUE(Send().ok());
  EXPECT_EQ(state().turns, 1);
}

TEST_F(RunnerServiceRulesTest, ACreateChatDuringASendIsRefused) {
  ASSERT_TRUE(Load().ok());
  ASSERT_TRUE(CreateChat().ok());
  Gate gate;
  script_.send_gate = &gate;
  grpc::Status sent;
  std::thread send([&] { sent = Send(); });
  gate.AwaitArrival();
  const grpc::Status refused = CreateChat();
  EXPECT_EQ(refused.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  EXPECT_EQ(refused.error_message(), "A Send is in progress.");
  gate.Open();
  send.join();
  EXPECT_TRUE(sent.ok()) << "the chat goes on: " << sent.error_message();
  EXPECT_EQ(script_.chats_at_create, 0) << "no chat was created meanwhile";
  EXPECT_EQ(state().turns, 1);
}

TEST_F(RunnerServiceRulesTest, AFailedCreateChatLeavesNoChat) {
  ASSERT_TRUE(Load().ok());
  ASSERT_TRUE(CreateChat().ok());
  ASSERT_TRUE(Send().ok());
  protocol::ChatOptions options = Options();
  options.set_max_completion_tokens(0);
  EXPECT_EQ(CreateChat(options).error_code(),
            grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(script_.chats, 0) << "the old chat is closed all the same";
  EXPECT_EQ(Send().error_message(), "No chat exists.");
  EXPECT_EQ(ReadCaptures(1).error_code(), grpc::StatusCode::NOT_FOUND);
  const grpc::Status again = CreateChat();
  EXPECT_TRUE(again.ok()) << again.error_message();
  EXPECT_TRUE(Send().ok());
}

// A fixed chat option out of the range the protocol gives it.
struct OutOfRange {
  std::string_view field;
  std::function<void(protocol::ChatOptions&)> set;
};

class ChatOptionsRangeTest : public RunnerServiceRulesTest,
                             public testing::WithParamInterface<OutOfRange> {};

TEST_P(ChatOptionsRangeTest, IsRefused) {
  ASSERT_TRUE(Load().ok());
  protocol::ChatOptions options = Options();
  GetParam().set(options);
  const grpc::Status refused = CreateChat(options);
  EXPECT_EQ(refused.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(refused.error_message().rfind(
                absl::StrCat("ChatOptions.", GetParam().field, " is "), 0),
            0u)
      << refused.error_message();
  EXPECT_TRUE(CreateChat().ok()) << "a refusal is not final";
}

INSTANTIATE_TEST_SUITE_P(
    EachField, ChatOptionsRangeTest,
    testing::Values(
        OutOfRange{
            "max_completion_tokens",
            [](protocol::ChatOptions& o) { o.set_max_completion_tokens(0); }},
        OutOfRange{"temperature",
                   [](protocol::ChatOptions& o) { o.set_temperature(-0.5); }},
        OutOfRange{
            "temperature",
            [](protocol::ChatOptions& o) { o.set_temperature(std::nanf("")); }},
        OutOfRange{"top_p", [](protocol::ChatOptions& o) { o.set_top_p(1.5); }},
        OutOfRange{"top_p",
                   [](protocol::ChatOptions& o) { o.set_top_p(-0.1f); }},
        OutOfRange{"top_k", [](protocol::ChatOptions& o) { o.set_top_k(0); }}));

}  // namespace
}  // namespace model_debugger::runner::service
// NOLINTEND
