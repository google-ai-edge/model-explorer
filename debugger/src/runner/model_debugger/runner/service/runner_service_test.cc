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

#include <atomic>
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
#include "model_debugger/runner/llm/litert_lm/litert_lm_runtime.h"
#include "model_debugger/runner/testing/test_utils.h"
#include "model_debugger/v1/runner.grpc.pb.h"
#include "model_debugger/v1/runner.pb.h"
#include "openssl/sha.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

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

// The SHA-256 of `bytes`, lower-case hexadecimal, as ListCaptures gives it.
std::string Sha256Hex(std::string_view bytes) {
  uint8_t digest[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), digest);
  return absl::BytesToHexString(
      std::string_view(reinterpret_cast<const char*>(digest), sizeof(digest)));
}

v1::SendRequest UserTurn(std::string text) {
  v1::SendRequest request;
  request.mutable_message()->set_role(v1::ChatMessage::USER);
  request.mutable_message()->set_content(std::move(text));
  return request;
}

// One server and one loaded, chatting runner for every test: loading takes
// seconds, and the service holds one model and one chat anyway. Tests run in
// declaration order and each leaves the chat usable, except the last.
//
// The first test that needs it builds it; gtest tears it down after the last
// test, before the process's own statics go: shutting the server down waits
// for a turn the runtime is still stopping, which must find the runtime whole.
struct Fixture {
  // Declared so that the server, which calls the service, goes first.
  std::unique_ptr<RunnerService> service;
  std::filesystem::path store;
  std::unique_ptr<v1::Runner::Stub> stub;
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
      fresh.stub = v1::Runner::NewStub(grpc::CreateChannel(
          Address(fresh.port), grpc::InsecureChannelCredentials()));
      fixture = std::move(fresh);
    }
    fixture_ = &*fixture;
  }

  // Where the server listens.
  static std::string Address(int port) {
    return "127.0.0.1:" + std::to_string(port);
  }

  v1::Runner::Stub& stub() { return *fixture_->stub; }
  int port() { return fixture_->port; }
  const std::filesystem::path& store() { return fixture_->store; }
  State state() { return fixture_->service->GetState(); }

  // The directory a turn's capture is filed in, beside the store.
  std::filesystem::path TurnDir(int turn) {
    return store().parent_path() / ("run." + std::to_string(getpid())) /
           ("turn-" + std::to_string(turn));
  }

  // ListCaptures for `turn`: its status, and the listing into `files`.
  grpc::StatusCode List(int turn, v1::ListCapturesResponse* files = nullptr) {
    grpc::ClientContext context;
    v1::ListCapturesRequest request;
    request.set_turn(turn);
    v1::ListCapturesResponse ignored;
    return stub()
        .ListCaptures(&context, request, files != nullptr ? files : &ignored)
        .error_code();
  }

  // ReadCapture of `path` in `turn`, to the end: its status, and the bytes
  // into `bytes`.
  grpc::StatusCode Read(int turn, const std::string& path,
                        std::string* bytes = nullptr) {
    grpc::ClientContext context;
    v1::ReadCaptureRequest request;
    request.set_turn(turn);
    request.set_path(path);
    std::unique_ptr<grpc::ClientReader<v1::CaptureData>> reader =
        stub().ReadCapture(&context, request);
    v1::CaptureData data;
    while (reader->Read(&data)) {
      if (bytes != nullptr) bytes->append(data.data());
    }
    return reader->Finish().error_code();
  }

  // Uploads `bytes` under `sha256`, announcing `size`, in 2-byte pieces;
  // without a header if `header` is false.
  grpc::Status Upload(std::string_view sha256, int64_t size,
                      std::string_view bytes, bool header = true) {
    grpc::ClientContext context;
    v1::UploadModelResponse response;
    std::unique_ptr<grpc::ClientWriter<v1::UploadModelRequest>> writer =
        stub().UploadModel(&context, &response);
    v1::UploadModelRequest request;
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
    std::vector<v1::Chunk> chunks;
    std::optional<v1::Completion> completion;
  };

  // Sends one turn and returns its status, the chunks and the completion.
  Reply Send(const std::string& text) {
    grpc::ClientContext context;
    std::unique_ptr<grpc::ClientReader<v1::SendResponse>> reader =
        stub().Send(&context, UserTurn(text));
    Reply reply;
    v1::SendResponse response;
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
  v1::RunnerInfo info;
  ASSERT_TRUE(stub().GetInfo(&context, v1::GetInfoRequest(), &info).ok());
  EXPECT_EQ(info.protocol_version(), "1.0.0");
  EXPECT_EQ(info.host().platform(), kPlatform);
  EXPECT_GT(info.host().memory_bytes(), 0);
  EXPECT_EQ(info.runtime().name(), v1::RuntimeInfo::LITERT_LM);
}

TEST_F(RunnerServiceTest, Step1AnotherClientIsRefused) {
  {
    grpc::ClientContext context;
    v1::RunnerInfo info;
    ASSERT_TRUE(stub().GetInfo(&context, v1::GetInfoRequest(), &info).ok());
  }
  // A subchannel pool of its own: another connection, from another port.
  grpc::ChannelArguments arguments;
  arguments.SetInt(GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL, 1);
  std::unique_ptr<v1::Runner::Stub> other =
      v1::Runner::NewStub(grpc::CreateCustomChannel(
          Address(port()), grpc::InsecureChannelCredentials(), arguments));
  {
    grpc::ClientContext context;
    v1::RunnerInfo info;
    EXPECT_EQ(
        other->GetInfo(&context, v1::GetInfoRequest(), &info).error_code(),
        grpc::StatusCode::FAILED_PRECONDITION);
  }
  grpc::ClientContext context;
  v1::RunnerInfo info;
  EXPECT_TRUE(stub().GetInfo(&context, v1::GetInfoRequest(), &info).ok())
      << "the first client is still served";
}

TEST_F(RunnerServiceTest, Step2LoadRefusesAModelTheStoreDoesNotHave) {
  grpc::ClientContext context;
  v1::LoadRequest request;
  request.set_sha256(std::string(64, '0'));
  request.set_device(DeviceFromEnvironment());
  request.set_context_length(1024);
  v1::LoadResponse response;
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
  {
    grpc::ClientContext context;
    v1::UploadModelResponse response;
    std::unique_ptr<grpc::ClientWriter<v1::UploadModelRequest>> writer =
        stub().UploadModel(&context, &response);
    v1::UploadModelRequest request;
    request.mutable_header()->set_sha256(std::string(kHelloSha256));
    request.mutable_header()->set_size((1 << 20) + 1);
    writer->Write(request);
    request.Clear();
    request.set_data(std::string((1 << 20) + 1, 'x'));
    writer->Write(request);
    writer->WritesDone();
    EXPECT_EQ(writer->Finish().error_code(), grpc::StatusCode::INVALID_ARGUMENT)
        << "a data message over 1 MiB";
  }
  EXPECT_FALSE(std::filesystem::exists(
      store() / (std::string(kHelloSha256) + ".litertlm")));
  for (const auto& entry : std::filesystem::directory_iterator(store())) {
    EXPECT_FALSE(
        absl::StrContains(entry.path().filename().string(), ".litertlm.part"))
        << entry.path();
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
  v1::LoadRequest request;
  request.set_sha256(std::string(kSha256));
  request.set_device(DeviceFromEnvironment());
  request.set_context_length(1024);
  // Two Loads at once: one loads, the other is refused, and the state
  // answers at once while the model loads.
  grpc::Status statuses[2];
  v1::LoadResponse responses[2];
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
    const absl::Time start = absl::Now();
    state();
    EXPECT_LT(absl::Now() - start, absl::Milliseconds(500))
        << "GetState waited for the Load";
    absl::SleepFor(absl::Milliseconds(10));
  }
  for (std::thread& load : loads) load.join();
  ASSERT_NE(statuses[0].ok(), statuses[1].ok())
      << statuses[0].error_message() << " / " << statuses[1].error_message();
  const int loaded = statuses[0].ok() ? 0 : 1;
  EXPECT_EQ(statuses[1 - loaded].error_code(),
            grpc::StatusCode::FAILED_PRECONDITION);
  EXPECT_FALSE(responses[loaded].stop_tokens().empty());

  v1::LoadResponse response;
  grpc::ClientContext again;
  EXPECT_EQ(stub().Load(&again, request, &response).error_code(),
            grpc::StatusCode::FAILED_PRECONDITION);
}

TEST_F(RunnerServiceTest, Step4SendBeforeTheChatIsRefused) {
  EXPECT_EQ(Send("Hi").status.error_code(),
            grpc::StatusCode::FAILED_PRECONDITION);
}

TEST_F(RunnerServiceTest, Step5CreateChatCreatesTheOneChatOnce) {
  v1::CreateChatRequest request;
  request.mutable_options()->set_max_completion_tokens(48);
  request.mutable_options()->set_top_k(1);
  v1::CreateChatResponse response;
  {
    grpc::ClientContext context;
    const grpc::Status status = stub().CreateChat(&context, request, &response);
    ASSERT_TRUE(status.ok()) << status.error_message();
  }
  grpc::ClientContext again;
  EXPECT_EQ(stub().CreateChat(&again, request, &response).error_code(),
            grpc::StatusCode::FAILED_PRECONDITION);
}

TEST_F(RunnerServiceTest, Step5AnInvalidTurnLeavesTheChatAsItWas) {
  v1::SendRequest request = UserTurn("I answer myself.");
  request.mutable_message()->set_role(v1::ChatMessage::ASSISTANT);
  grpc::ClientContext context;
  std::unique_ptr<grpc::ClientReader<v1::SendResponse>> reader =
      stub().Send(&context, request);
  v1::SendResponse response;
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
  for (const v1::Chunk& chunk : reply.chunks) streamed += chunk.content();
  EXPECT_EQ(reply.completion->content(), streamed);
  EXPECT_EQ(reply.completion->model_debugger().turn(), 1);
  const State now = state();
  EXPECT_FALSE(now.client.empty());
  EXPECT_EQ(now.model_sha256, kSha256);
  EXPECT_EQ(now.turns, 1);
  EXPECT_FALSE(now.chat_over);
}

TEST_F(RunnerServiceTest, Step7CapturesOfTheTurnCanBeListedAndRead) {
  // A file of several messages, filed with the turn's own.
  const std::string large(5 << 19, 'x');
  std::ofstream(TurnDir(1) / "large.bin", std::ios::binary) << large;

  grpc::ClientContext context;
  v1::ListCapturesRequest request;
  request.set_turn(1);
  v1::ListCapturesResponse listing;
  ASSERT_TRUE(stub().ListCaptures(&context, request, &listing).ok());
  bool tokens = false;
  bool large_listed = false;
  for (const v1::CaptureFile& entry : listing.files()) {
    if (entry.path() == "generated_tokens.jsonl") tokens = true;
    if (entry.path() == "large.bin") large_listed = true;
    std::string bytes;
    ASSERT_EQ(Read(1, entry.path(), &bytes), grpc::StatusCode::OK)
        << entry.path();
    EXPECT_EQ(static_cast<int64_t>(bytes.size()), entry.size()) << entry.path();
    EXPECT_EQ(Sha256Hex(bytes), entry.sha256()) << entry.path();
  }
  EXPECT_TRUE(tokens) << "the turn's tokens are filed with its tensors";
  EXPECT_TRUE(large_listed);

  EXPECT_EQ(Read(1, "."), grpc::StatusCode::NOT_FOUND) << "a directory";
  EXPECT_EQ(Read(1, ".."), grpc::StatusCode::NOT_FOUND) << "a directory";
  EXPECT_EQ(Read(1, "../turn-1/large.bin"), grpc::StatusCode::INVALID_ARGUMENT)
      << "a path out of the turn's directory and back";
  EXPECT_EQ(Read(1, "../../models/" + std::string(kSha256) + ".litertlm"),
            grpc::StatusCode::INVALID_ARGUMENT)
      << "a path out of the turn's directory";
  EXPECT_EQ(Read(1, (TurnDir(1) / "large.bin").string()),
            grpc::StatusCode::INVALID_ARGUMENT)
      << "an absolute path";

  grpc::ClientContext missing;
  v1::ListCapturesRequest none;
  none.set_turn(7);
  EXPECT_EQ(stub().ListCaptures(&missing, none, &listing).error_code(),
            grpc::StatusCode::NOT_FOUND);
}

TEST_F(RunnerServiceTest, Step7AReleasedCaptureIsGone) {
  v1::ReleaseCapturesRequest release;
  release.set_turn(1);
  v1::ReleaseCapturesResponse released;
  {
    grpc::ClientContext context;
    ASSERT_TRUE(stub().ReleaseCaptures(&context, release, &released).ok());
  }
  EXPECT_FALSE(std::filesystem::exists(TurnDir(1))) << "deleted from disk";
  EXPECT_EQ(List(1), grpc::StatusCode::NOT_FOUND);
  EXPECT_EQ(Read(1, "generated_tokens.jsonl"), grpc::StatusCode::NOT_FOUND);
  grpc::ClientContext again;
  EXPECT_EQ(stub().ReleaseCaptures(&again, release, &released).error_code(),
            grpc::StatusCode::NOT_FOUND);
}

TEST_F(RunnerServiceTest, Step8ASecondTurnIsFiledApart) {
  const Reply reply = Send("And one more?");
  ASSERT_TRUE(reply.status.ok()) << reply.status.error_message();
  ASSERT_TRUE(reply.completion.has_value());
  EXPECT_EQ(reply.completion->model_debugger().turn(), 2);
  // Creating and destroying a second RunnerService in the same process keeps
  // its own run.<pid>.<id> directory and leaves the first service's captures
  // untouched.
  {
    absl::StatusOr<std::unique_ptr<RunnerService>> second =
        RunnerService::Create(std::make_unique<llm::LiteRtLmRuntime>());
    ASSERT_TRUE(second.ok()) << second.status();
  }
  v1::ListCapturesResponse files;
  ASSERT_EQ(List(2, &files), grpc::StatusCode::OK);
  EXPECT_FALSE(files.files().empty());
  EXPECT_EQ(Read(2, "generated_tokens.jsonl"), grpc::StatusCode::OK);
}

TEST_F(RunnerServiceTest, Step9CancellingSendEndsTheChat) {
  grpc::ClientContext context;
  std::unique_ptr<grpc::ClientReader<v1::SendResponse>> reader =
      stub().Send(&context, UserTurn("Count from one to one thousand."));
  v1::SendResponse response;
  ASSERT_TRUE(reader->Read(&response)) << "a first chunk";
  context.TryCancel();
  while (reader->Read(&response)) {
  }
  EXPECT_EQ(reader->Finish().error_code(), grpc::StatusCode::CANCELLED);

  // The client's cancellation returns at once; the runner is still stopping
  // the model. Until it has, a Send is refused as one in progress.
  const absl::Time deadline = absl::Now() + absl::Seconds(30);
  Reply again = Send("Again?");
  while (again.status.error_message() == "A Send is in progress." &&
         absl::Now() < deadline) {
    absl::SleepFor(absl::Milliseconds(50));
    again = Send("Again?");
  }
  EXPECT_EQ(again.status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  EXPECT_EQ(again.status.error_message(),
            "The chat is over; generation on this connection ended with a "
            "cancelled or failed turn.");
  EXPECT_TRUE(state().chat_over);
  EXPECT_EQ(state().turns, 2);
  EXPECT_EQ(List(3), grpc::StatusCode::NOT_FOUND) << "a cancelled turn";
  EXPECT_EQ(List(2), grpc::StatusCode::OK) << "earlier turns stay readable";
}

}  // namespace
}  // namespace model_debugger::runner::service
