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

// The Runner service end to end: a server in this process, a client stub
// against it, LiteRT-LM underneath. Skipped unless RUNNER_MODEL names a
// .litertlm file. The GPU variant (testing/model_test.bzl) sets
// RUNNER_BACKEND=gpu.

#include "model_debugger/runner/service/runner_service.h"

#include <unistd.h>

// NOLINTBEGIN(build/c++11)
// NOLINTBEGIN(build/c++17)
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "grpcpp/grpcpp.h"
#include "model_debugger/protocol/runner.grpc.pb.h"
#include "model_debugger/protocol/runner.pb.h"
#include "model_debugger/runner/llm/litert_lm/litert_lm_runtime.h"
#include "model_debugger/runner/testing/test_utils.h"
#include "openssl/sha.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"

namespace model_debugger::runner::service {
namespace {

// The model store names a file by sha256; the service looks it up by name and
// trusts the name, so the test's store links the model under one.
constexpr std::string_view kSha256 =
    "9f3a7c1e5b2d4a6f8e0c1b3d5f7a9c2e4b6d8f0a1c3e5b7d9f2a4c6e8b0d1f3a";

// Five bytes and their SHA-256: a file small enough to upload in a test.
constexpr std::string_view kHello = "hello";
constexpr std::string_view kHelloSha256 =
    "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

// The SHA-256 of `bytes`, lower-case hexadecimal, as ReadCaptures gives it.
std::string Sha256Hex(std::string_view bytes) {
  uint8_t digest[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), digest);
  return absl::BytesToHexString(
      std::string_view(reinterpret_cast<const char*>(digest), sizeof(digest)));
}

protocol::SendRequest UserTurn(std::string text) {
  protocol::SendRequest request;
  request.mutable_message()->set_role(protocol::ChatMessage::USER);
  request.mutable_message()->set_content(std::move(text));
  return request;
}

// A greedy chat of short replies that starts with the system message
// `system`.
protocol::CreateChatRequest ChatWithSystem(std::string system) {
  protocol::CreateChatRequest request;
  protocol::ChatMessage& message = *request.add_messages();
  message.set_role(protocol::ChatMessage::SYSTEM);
  message.set_content(std::move(system));
  request.mutable_options()->set_max_completion_tokens(48);
  request.mutable_options()->set_top_k(1);
  return request;
}

// Returns the texts of the prompt the turn of `debugger` fed, joined: the
// tokens up to the first decode step, which feeds the prompt's last.
std::string PromptText(const protocol::ModelDebugger& debugger) {
  std::string text;
  for (const protocol::FedToken& token : debugger.fed_tokens()) {
    text += token.text();
    if (token.phase() == protocol::FedToken::DECODE) break;
  }
  return text;
}

// One server and one loaded, chatting runner for every test: loading takes
// seconds, and the service holds one model anyway. Tests run in declaration
// order and each leaves the chat usable, but the one that cancels a turn,
// after which the next replaces the chat.
//
// The first test that needs it builds it; gtest tears it down after the last
// test, before the process's own statics go: shutting the server down waits
// for a turn the runtime is still stopping, which must find the runtime whole.
struct Fixture {
  // Declared so that the server, which calls the service, goes first.
  std::unique_ptr<RunnerService> service;
  std::filesystem::path store;
  std::unique_ptr<protocol::Runner::Stub> stub;
  std::unique_ptr<grpc::Server> server;
  int port = 0;  // The server's, on 127.0.0.1.

  Fixture() = default;
  Fixture(Fixture&&) = default;
  Fixture& operator=(Fixture&&) = default;
  ~Fixture() {
    if (server != nullptr) server->Shutdown();
  }
};

std::optional<Fixture>& SharedFixture() {
  static std::optional<Fixture>* const fixture = new std::optional<Fixture>;
  return *fixture;
}

class FixtureEnvironment : public testing::Environment {
 public:
  void TearDown() override { SharedFixture().reset(); }
};

const testing::Environment* const kEnvironment =
    testing::AddGlobalTestEnvironment(new FixtureEnvironment);

class RunnerServiceTest : public testing::Test {
 protected:
  void SetUp() override {
    std::optional<Fixture>& fixture = SharedFixture();
    const char* model = std::getenv("RUNNER_MODEL");
    if (model == nullptr) GTEST_SKIP() << "RUNNER_MODEL is not set";
    if (!fixture) {
      Fixture fresh;
      // The service keeps models in its cache directory: one of this run's
      // own, with the model linked into it under its name.
      fresh.store = UseTestCacheDirectory() / "models";
      std::filesystem::create_directories(fresh.store);
      std::filesystem::create_symlink(
          model, fresh.store / (std::string(kSha256) + ".litertlm"));
      absl::StatusOr<std::unique_ptr<RunnerService>> service =
          RunnerService::Create(std::make_unique<llm::LiteRtLmRuntime>());
      ASSERT_TRUE(service.ok()) << service.status();
      fresh.service = *std::move(service);
      grpc::ServerBuilder builder;
      builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(),
                               &fresh.port);
      builder.RegisterService(fresh.service.get());
      fresh.server = builder.BuildAndStart();
      ASSERT_NE(fresh.server, nullptr);
      fresh.stub = protocol::Runner::NewStub(grpc::CreateChannel(
          Address(fresh.port), grpc::InsecureChannelCredentials()));
      fixture = std::move(fresh);
    }
    fixture_ = &*fixture;
  }

  // Where the server listens.
  static std::string Address(int port) {
    return "127.0.0.1:" + std::to_string(port);
  }

  protocol::Runner::Stub& stub() { return *fixture_->stub; }
  int port() { return fixture_->port; }
  const std::filesystem::path& store() { return fixture_->store; }
  State state() { return fixture_->service->GetState(); }

  // The directory a turn's capture is filed in, beside the store.
  std::filesystem::path TurnDir(int turn) {
    return store().parent_path() / ("run." + std::to_string(getpid())) /
           ("turn-" + std::to_string(turn));
  }

  // One file of a ReadCaptures stream, as it arrived.
  struct CapturedFile {
    std::string path;
    int64_t size = 0;
    std::string bytes;
    size_t largest_data = 0;  // The most bytes one data message carried.
    std::string sha256;       // Empty until the file ended.
  };

  // ReadCaptures for `turn`, to the end: its status, and the files in the
  // order they came into `files`. Fails the test if a message comes out of
  // order: data or sha256 outside a file, or a file before the last ended.
  grpc::StatusCode ReadCaptures(int turn,
                                std::vector<CapturedFile>* files = nullptr) {
    grpc::ClientContext context;
    protocol::ReadCapturesRequest request;
    request.set_turn(turn);
    std::unique_ptr<grpc::ClientReader<protocol::CaptureChunk>> reader =
        stub().ReadCaptures(&context, request);
    std::vector<CapturedFile> received;
    bool open = false;  // A file started and has not ended.
    protocol::CaptureChunk chunk;
    while (reader->Read(&chunk)) {
      switch (chunk.part_case()) {
        case protocol::CaptureChunk::kFile:
          EXPECT_FALSE(open) << "a file before the last one ended";
          received.emplace_back();
          received.back().path = chunk.file().path();
          received.back().size = chunk.file().size();
          open = true;
          break;
        case protocol::CaptureChunk::kData:
          EXPECT_TRUE(open) << "data outside a file";
          if (!open) break;
          received.back().bytes += chunk.data();
          received.back().largest_data =
              std::max(received.back().largest_data, chunk.data().size());
          break;
        case protocol::CaptureChunk::kSha256:
          EXPECT_TRUE(open) << "a sha256 outside a file";
          if (!open) break;
          received.back().sha256 = chunk.sha256();
          open = false;
          break;
        default:
          ADD_FAILURE() << "an empty message";
      }
    }
    const grpc::StatusCode code = reader->Finish().error_code();
    if (code == grpc::StatusCode::OK) {
      EXPECT_FALSE(open) << "the stream ended mid-file";
    }
    if (files != nullptr) *files = std::move(received);
    return code;
  }

  // Uploads `bytes` under `sha256`, announcing `size`, in 2-byte pieces;
  // without a header if `header` is false.
  grpc::Status Upload(std::string_view sha256, int64_t size,
                      std::string_view bytes, bool header = true) {
    grpc::ClientContext context;
    protocol::UploadModelResponse response;
    std::unique_ptr<grpc::ClientWriter<protocol::UploadModelRequest>> writer =
        stub().UploadModel(&context, &response);
    protocol::UploadModelRequest request;
    if (header) {
      request.mutable_header()->set_sha256(std::string(sha256));
      request.mutable_header()->set_size(size);
      writer->Write(request);
    }
    for (size_t i = 0; i < bytes.size(); i += 2) {
      request.Clear();
      request.set_data(std::string(bytes.substr(i, 2)));
      if (!writer->Write(request)) break;
    }
    writer->WritesDone();
    return writer->Finish();
  }

  // What one Send returned: its status, the chunks and the completion.
  struct Reply {
    grpc::Status status;
    std::vector<protocol::Chunk> chunks;
    std::optional<protocol::Completion> completion;
  };

  // Sends one turn and returns its status, the chunks and the completion.
  Reply Send(const std::string& text) {
    grpc::ClientContext context;
    std::unique_ptr<grpc::ClientReader<protocol::SendResponse>> reader =
        stub().Send(&context, UserTurn(text));
    Reply reply;
    protocol::SendResponse response;
    while (reader->Read(&response)) {
      if (response.has_chunk()) reply.chunks.push_back(response.chunk());
      if (response.has_completion()) reply.completion = response.completion();
    }
    reply.status = reader->Finish();
    return reply;
  }

 private:
  Fixture* fixture_ = nullptr;
};

TEST_F(RunnerServiceTest, Step1GetInfoNamesTheRunner) {
  grpc::ClientContext context;
  protocol::RunnerInfo info;
  ASSERT_TRUE(stub().GetInfo(&context, protocol::GetInfoRequest(), &info).ok());
  EXPECT_EQ(info.protocol_version(),
            protocol::RunnerInfo::descriptor()->file()->options().GetExtension(
                protocol::protocol_version));
  EXPECT_EQ(info.host().platform(), kPlatform);
  EXPECT_GT(info.host().memory_bytes(), 0);
  EXPECT_EQ(info.runtime().name(), protocol::RuntimeInfo::LITERT_LM);
}

TEST_F(RunnerServiceTest, Step1AnotherConnectionIsServed) {
  {
    grpc::ClientContext context;
    protocol::RunnerInfo info;
    ASSERT_TRUE(
        stub().GetInfo(&context, protocol::GetInfoRequest(), &info).ok());
  }
  // A subchannel pool of its own: another connection, from another port.
  grpc::ChannelArguments arguments;
  arguments.SetInt(GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL, 1);
  std::unique_ptr<protocol::Runner::Stub> other =
      protocol::Runner::NewStub(grpc::CreateCustomChannel(
          Address(port()), grpc::InsecureChannelCredentials(), arguments));
  grpc::ClientContext context;
  protocol::RunnerInfo info;
  EXPECT_TRUE(other->GetInfo(&context, protocol::GetInfoRequest(), &info).ok());
}

TEST_F(RunnerServiceTest, Step2LoadRefusesAModelTheStoreDoesNotHave) {
  grpc::ClientContext context;
  protocol::LoadRequest request;
  request.set_sha256(std::string(64, '0'));
  request.set_device(DeviceFromEnvironment());
  request.set_context_length(1024);
  protocol::LoadResponse response;
  EXPECT_EQ(stub().Load(&context, request, &response).error_code(),
            grpc::StatusCode::NOT_FOUND);
}

TEST_F(RunnerServiceTest, Step2UploadRefusesAStreamWithoutAHeader) {
  EXPECT_EQ(Upload(kHelloSha256, 5, kHello, /*header=*/false).error_code(),
            grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(RunnerServiceTest, Step2UploadKeepsNothingOfAFailedUpload) {
  EXPECT_EQ(Upload(kHelloSha256, 5, "hellp").error_code(),
            grpc::StatusCode::DATA_LOSS)
      << "bytes that do not hash to the sha256";
  EXPECT_EQ(Upload(kHelloSha256, 6, kHello).error_code(),
            grpc::StatusCode::DATA_LOSS)
      << "fewer bytes than announced";
  EXPECT_EQ(Upload(kHelloSha256, 4, kHello).error_code(),
            grpc::StatusCode::INVALID_ARGUMENT)
      << "more bytes than announced";
  EXPECT_EQ(Upload("not a sha256", 5, kHello).error_code(),
            grpc::StatusCode::INVALID_ARGUMENT);
  for (const auto& entry : std::filesystem::directory_iterator(store())) {
    EXPECT_FALSE(
        absl::StartsWith(entry.path().filename().string(), kHelloSha256))
        << entry.path() << " is left of a failed upload";
  }
}

TEST_F(RunnerServiceTest, Step2UploadPutsTheFileInTheStoreOnce) {
  ASSERT_TRUE(Upload(kHelloSha256, 5, kHello).ok());
  const std::filesystem::path file =
      store() / (std::string(kHelloSha256) + ".litertlm");
  ASSERT_TRUE(std::filesystem::is_regular_file(file));
  EXPECT_EQ(std::filesystem::file_size(file), 5u);
  EXPECT_TRUE(Upload(kHelloSha256, 5, "other").ok())
      << "already in the store: the stream is not read";
  EXPECT_EQ(std::filesystem::file_size(file), 5u);
}

TEST_F(RunnerServiceTest, Step3LoadLoadsTheOneModelOnce) {
  // The Load and the uploads of other models before had the store keep only
  // theirs, which may have deleted the link to the model.
  const std::filesystem::path link =
      store() / (std::string(kSha256) + ".litertlm");
  if (!std::filesystem::is_symlink(link)) {
    std::filesystem::create_symlink(std::getenv("RUNNER_MODEL"), link);
  }
  protocol::LoadRequest request;
  request.set_sha256(std::string(kSha256));
  request.set_device(DeviceFromEnvironment());
  request.set_context_length(1024);
  // Two Loads at once: one loads, the other is refused, and the state
  // answers at once while the model loads.
  grpc::Status statuses[2];
  protocol::LoadResponse responses[2];
  std::atomic<int> finished = 0;
  std::vector<std::thread> loads;
  for (int i = 0; i < 2; ++i) {
    loads.emplace_back([&, i] {
      grpc::ClientContext context;
      statuses[i] = stub().Load(&context, request, &responses[i]);
      ++finished;
    });
  }
  while (finished < 2) {
    const auto start = std::chrono::steady_clock::now();
    state();
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - start)
                  .count(),
              500)
        << "GetState waited for the Load";
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  for (std::thread& load : loads) load.join();
  ASSERT_NE(statuses[0].ok(), statuses[1].ok())
      << statuses[0].error_message() << " / " << statuses[1].error_message();
  const int loaded = statuses[0].ok() ? 0 : 1;
  EXPECT_EQ(statuses[1 - loaded].error_code(),
            grpc::StatusCode::FAILED_PRECONDITION);
  EXPECT_FALSE(responses[loaded].stop_tokens().empty());

  protocol::LoadResponse response;
  grpc::ClientContext again;
  EXPECT_EQ(stub().Load(&again, request, &response).error_code(),
            grpc::StatusCode::FAILED_PRECONDITION);
}

TEST_F(RunnerServiceTest, Step4SendBeforeTheChatIsRefused) {
  EXPECT_EQ(Send("Hi").status.error_code(),
            grpc::StatusCode::FAILED_PRECONDITION);
}

TEST_F(RunnerServiceTest, Step5CreateChatCreatesTheChat) {
  grpc::ClientContext context;
  protocol::CreateChatResponse response;
  const grpc::Status status = stub().CreateChat(
      &context, ChatWithSystem("You are a terse assistant."), &response);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(RunnerServiceTest, Step5AnInvalidTurnLeavesTheChatAsItWas) {
  protocol::SendRequest request = UserTurn("I answer myself.");
  request.mutable_message()->set_role(protocol::ChatMessage::ASSISTANT);
  grpc::ClientContext context;
  std::unique_ptr<grpc::ClientReader<protocol::SendResponse>> reader =
      stub().Send(&context, request);
  protocol::SendResponse response;
  while (reader->Read(&response)) {
  }
  EXPECT_EQ(reader->Finish().error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_FALSE(state().chat_over);
  EXPECT_EQ(state().turns, 0) << "a refused turn is no turn";
}

TEST_F(RunnerServiceTest, Step6ATurnStreamsAndIsFiled) {
  const Reply reply = Send("Say hello in one short sentence.");
  ASSERT_TRUE(reply.status.ok()) << reply.status.error_message();
  ASSERT_TRUE(reply.completion.has_value());
  std::string streamed;
  for (const protocol::Chunk& chunk : reply.chunks) streamed += chunk.content();
  EXPECT_EQ(reply.completion->content(), streamed);
  EXPECT_EQ(reply.completion->model_debugger().turn(), 1);
  const State now = state();
  EXPECT_FALSE(now.client.empty());
  EXPECT_EQ(now.model_sha256, kSha256);
  EXPECT_EQ(now.turns, 1);
  EXPECT_FALSE(now.chat_over);
}

TEST_F(RunnerServiceTest, Step7CapturesOfTheTurnStreamFileByFile) {
  // A file of several messages, filed with the turn's own.
  const std::string large(5 << 19, 'x');
  std::ofstream(TurnDir(1) / "large.bin", std::ios::binary) << large;

  std::vector<CapturedFile> files;
  ASSERT_EQ(ReadCaptures(1, &files), grpc::StatusCode::OK);
  std::vector<std::string> paths;
  for (const CapturedFile& file : files) {
    paths.push_back(file.path);
    EXPECT_EQ(static_cast<int64_t>(file.bytes.size()), file.size) << file.path;
    EXPECT_EQ(Sha256Hex(file.bytes), file.sha256) << file.path;
    EXPECT_LE(file.largest_data, size_t{1} << 20) << file.path;
    if (file.path == "large.bin") EXPECT_EQ(file.bytes, large);
  }
  EXPECT_TRUE(std::is_sorted(paths.begin(), paths.end()))
      << "a stable order: by name";
  EXPECT_NE(std::find(paths.begin(), paths.end(), "generated_tokens.jsonl"),
            paths.end())
      << "the turn's tokens are filed with its tensors";
  EXPECT_NE(std::find(paths.begin(), paths.end(), "large.bin"), paths.end());
  EXPECT_EQ(ReadCaptures(1), grpc::StatusCode::OK) << "read again, unchanged";

  EXPECT_EQ(ReadCaptures(7), grpc::StatusCode::NOT_FOUND) << "no such turn";
  EXPECT_EQ(ReadCaptures(0), grpc::StatusCode::NOT_FOUND) << "no such turn";
}

TEST_F(RunnerServiceTest, Step7AReleasedCaptureIsGone) {
  protocol::ReleaseCapturesRequest release;
  release.set_turn(1);
  protocol::ReleaseCapturesResponse released;
  {
    grpc::ClientContext context;
    ASSERT_TRUE(stub().ReleaseCaptures(&context, release, &released).ok());
  }
  EXPECT_FALSE(std::filesystem::exists(TurnDir(1))) << "deleted from disk";
  EXPECT_EQ(ReadCaptures(1), grpc::StatusCode::NOT_FOUND);
  grpc::ClientContext again;
  EXPECT_EQ(stub().ReleaseCaptures(&again, release, &released).error_code(),
            grpc::StatusCode::NOT_FOUND);
}

TEST_F(RunnerServiceTest, Step8ASecondTurnIsFiledApart) {
  const Reply reply = Send("And one more?");
  ASSERT_TRUE(reply.status.ok()) << reply.status.error_message();
  ASSERT_TRUE(reply.completion.has_value());
  EXPECT_EQ(reply.completion->model_debugger().turn(), 2);
  std::vector<CapturedFile> files;
  ASSERT_EQ(ReadCaptures(2, &files), grpc::StatusCode::OK);
  EXPECT_FALSE(files.empty());
}

TEST_F(RunnerServiceTest, Step9CancellingSendEndsTheChat) {
  grpc::ClientContext context;
  std::unique_ptr<grpc::ClientReader<protocol::SendResponse>> reader =
      stub().Send(&context, UserTurn("Count from one to one thousand."));
  protocol::SendResponse response;
  ASSERT_TRUE(reader->Read(&response)) << "a first chunk";
  context.TryCancel();
  while (reader->Read(&response)) {
  }
  EXPECT_EQ(reader->Finish().error_code(), grpc::StatusCode::CANCELLED);

  // The client's cancellation returns at once; the runner is still stopping
  // the model. Until it has, a Send is refused as one in progress.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  Reply again = Send("Again?");
  while (again.status.error_message() == "A Send is in progress." &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    again = Send("Again?");
  }
  EXPECT_EQ(again.status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  EXPECT_EQ(again.status.error_message(),
            "The chat has ended with a cancelled or failed turn; a "
            "CreateChat starts another.");
  EXPECT_TRUE(state().chat_over);
  EXPECT_EQ(state().turns, 2);
  EXPECT_EQ(ReadCaptures(3), grpc::StatusCode::NOT_FOUND) << "a cancelled turn";
  EXPECT_EQ(ReadCaptures(2), grpc::StatusCode::OK)
      << "earlier turns stay readable";
}

TEST_F(RunnerServiceTest, Step10ANewChatReplacesTheOneThatIsOver) {
  {
    grpc::ClientContext context;
    protocol::CreateChatResponse response;
    const grpc::Status status = stub().CreateChat(
        &context, ChatWithSystem("You are a pirate."), &response);
    ASSERT_TRUE(status.ok()) << status.error_message();
  }
  EXPECT_FALSE(state().chat_over);
  EXPECT_EQ(state().turns, 0);
  EXPECT_EQ(ReadCaptures(2), grpc::StatusCode::NOT_FOUND)
      << "the old chat's captures are deleted";
  EXPECT_FALSE(std::filesystem::exists(TurnDir(2)));

  const Reply reply = Send("Name one color.");
  ASSERT_TRUE(reply.status.ok()) << reply.status.error_message();
  ASSERT_TRUE(reply.completion.has_value());
  const protocol::ModelDebugger& debugger = reply.completion->model_debugger();
  EXPECT_EQ(debugger.turn(), 1) << "the new chat's turns count from 1";
  ASSERT_FALSE(debugger.fed_tokens().empty());
  EXPECT_EQ(debugger.fed_tokens(0).position(), 0)
      << "the new chat's context starts empty";
  const std::string prompt = PromptText(debugger);
  EXPECT_NE(prompt.find("pirate"), std::string::npos) << prompt;
  for (const char* old :
       {"terse", "Say hello", "And one more?", "Count from"}) {
    EXPECT_EQ(prompt.find(old), std::string::npos)
        << "the old chat's \"" << old << "\" is not in the new one: " << prompt;
  }
  EXPECT_EQ(debugger.context_token_count(),
            reply.completion->usage().total_tokens())
      << "the context holds this turn alone";

  std::vector<CapturedFile> files;
  ASSERT_EQ(ReadCaptures(1, &files), grpc::StatusCode::OK);
  int tensors = 0;
  for (const CapturedFile& file : files) {
    if (absl::EndsWith(file.path, ".safetensors")) ++tensors;
    if (file.path == "generated_tokens.jsonl") {
      EXPECT_FALSE(file.bytes.empty()) << "the new chat's own lines";
    }
  }
  EXPECT_GT(tensors, 0) << "the new chat's capture directory is the one read";
}

}  // namespace
}  // namespace model_debugger::runner::service
// NOLINTEND
