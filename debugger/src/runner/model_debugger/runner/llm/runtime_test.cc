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

// The contract of llm::Runtime, run on LiteRT-LM when RUNNER_MODEL names a
// .litertlm file; skipped otherwise. The GPU variant (testing/model_test.bzl)
// sets RUNNER_BACKEND=gpu.
// A second runtime would be tested here too.

#include "model_debugger/runner/llm/runtime.h"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "google/protobuf/struct.pb.h"
#include "gtest/gtest.h"
#include "model_debugger/runner/llm/litert_lm/litert_lm_runtime.h"
#include "model_debugger/runner/testing/test_utils.h"
#include "model_debugger/v1/runner.pb.h"

namespace model_debugger::runner::llm {
namespace {

v1::ChatMessage Message(v1::ChatMessage::Role role, std::string text) {
  v1::ChatMessage message;
  message.set_role(role);
  message.set_content(std::move(text));
  return message;
}

v1::ChatMessage User(std::string text) {
  return Message(v1::ChatMessage::USER, std::move(text));
}

// Returns the texts of the prompt tokens in `debugger`, joined: the prompt as
// the model read it.
std::string PromptText(const v1::ModelDebugger& debugger) {
  std::string text;
  for (const v1::PromptToken& token : debugger.prompt_tokens()) {
    text += token.text();
  }
  return text;
}

// One runtime and one loaded model, shared by the tests: loading takes
// seconds. The runtime is declared first so that the model, which must not
// outlive it, is destroyed first.
struct Loaded {
  std::unique_ptr<Runtime> runtime;
  std::unique_ptr<Model> model;
  std::filesystem::path model_path;
  std::filesystem::path data_dir;
};

class RuntimeTest : public testing::Test {
 protected:
  void SetUp() override {
    static absl::NoDestructor<std::optional<Loaded>> loaded;
    const char* model = std::getenv("RUNNER_MODEL");
    if (model == nullptr) GTEST_SKIP() << "RUNNER_MODEL is not set";
    if (!*loaded) {
      Loaded fresh;
      fresh.runtime = std::make_unique<LiteRtLmRuntime>();
      fresh.model_path = model;
      fresh.data_dir = TempDir("data");
      absl::StatusOr<std::unique_ptr<Model>> loaded_model =
          fresh.runtime->Load(fresh.model_path, fresh.data_dir, Request());
      ASSERT_TRUE(loaded_model.ok()) << loaded_model.status();
      fresh.model = *std::move(loaded_model);
      *loaded = std::move(fresh);
    }
    loaded_ = &**loaded;
  }

  static v1::LoadRequest Request() {
    v1::LoadRequest request;
    request.set_device(DeviceFromEnvironment());
    request.set_context_length(1024);
    return request;
  }

  Runtime& runtime() { return *loaded_->runtime; }
  Model& model() { return *loaded_->model; }
  const std::filesystem::path& model_path() { return loaded_->model_path; }
  const std::filesystem::path& data_dir() { return loaded_->data_dir; }

  // Returns a chat that always picks the likeliest token, so that a test is
  // the same each time.
  std::unique_ptr<Chat> CreateChat(int max_completion_tokens,
                                   std::vector<v1::ChatMessage> messages = {},
                                   bool thinking = false) {
    v1::CreateChatRequest request;
    for (v1::ChatMessage& message : messages) {
      *request.add_messages() = std::move(message);
    }
    request.mutable_options()->set_max_completion_tokens(max_completion_tokens);
    request.mutable_options()->set_top_k(1);
    if (thinking) {
      (*request.mutable_options()
            ->mutable_runtime_options()
            ->mutable_fields())["thinking"]
          .set_bool_value(true);
    }
    absl::StatusOr<std::unique_ptr<Chat>> chat = model().CreateChat(request);
    EXPECT_TRUE(chat.ok()) << chat.status();
    return chat.ok() ? *std::move(chat) : nullptr;
  }

 private:
  Loaded* loaded_ = nullptr;
};

// ---- Load

TEST_F(RuntimeTest, LoadRefusesAMissingModel) {
  EXPECT_TRUE(absl::IsNotFound(
      runtime()
          .Load("/no/such/model.litertlm", data_dir(), Request())
          .status()));
}

TEST_F(RuntimeTest, LoadRefusesAMissingDataDirectory) {
  EXPECT_TRUE(absl::IsFailedPrecondition(
      runtime().Load(model_path(), "/no/such/directory", Request()).status()));
}

TEST_F(RuntimeTest, LoadRefusesAContextLengthThatIsNotPositive) {
  v1::LoadRequest request = Request();
  request.set_context_length(0);
  EXPECT_TRUE(absl::IsInvalidArgument(
      runtime().Load(model_path(), data_dir(), request).status()));
}

TEST_F(RuntimeTest, LoadRefusesAnUnspecifiedDevice) {
  v1::LoadRequest request = Request();
  request.set_device(v1::DEVICE_UNSPECIFIED);
  EXPECT_TRUE(absl::IsInvalidArgument(
      runtime().Load(model_path(), data_dir(), request).status()));
}

// ---- Info

TEST_F(RuntimeTest, RuntimeInfoNamesTheRuntimeAndItsOptions) {
  const v1::RuntimeInfo& info = runtime().Info();
  EXPECT_EQ(info.name(), v1::RuntimeInfo::LITERT_LM);
  EXPECT_EQ(info.version().size(), 40u) << "the LiteRT-LM commit, whole";
  EXPECT_NE(info.load_options_schema().find("cpuThreads"), std::string::npos);
  EXPECT_NE(info.chat_options_schema().find("thinking"), std::string::npos);
}

TEST_F(RuntimeTest, ModelInfoHasTheStopTokens) {
  const v1::LoadResponse& info = model().Info();
  ASSERT_FALSE(info.stop_tokens().empty());
  for (const v1::StopToken& stop : info.stop_tokens()) {
    EXPECT_TRUE(stop.has_ids() || !stop.text().empty());
    if (stop.has_ids()) EXPECT_FALSE(stop.ids().ids().empty());
  }
}

// ---- CreateChat

TEST_F(RuntimeTest, CreateChatRefusesMaxCompletionTokensThatIsNotPositive) {
  v1::CreateChatRequest request;
  request.mutable_options()->set_max_completion_tokens(0);
  EXPECT_TRUE(absl::IsInvalidArgument(model().CreateChat(request).status()));
}

TEST_F(RuntimeTest, CreateChatRefusesASystemMessageThatIsNotFirst) {
  v1::CreateChatRequest request;
  request.mutable_options()->set_max_completion_tokens(8);
  *request.add_messages() = User("Hi");
  *request.add_messages() = Message(v1::ChatMessage::SYSTEM, "Be terse.");
  EXPECT_TRUE(absl::IsInvalidArgument(model().CreateChat(request).status()));
}

TEST_F(RuntimeTest, CreateChatRefusesAChatOptionTheRuntimeDoesNotHave) {
  v1::CreateChatRequest request;
  request.mutable_options()->set_max_completion_tokens(8);
  (*request.mutable_options()
        ->mutable_runtime_options()
        ->mutable_fields())["mood"]
      .set_string_value("cheerful");
  EXPECT_TRUE(absl::IsInvalidArgument(model().CreateChat(request).status()));
}

TEST_F(RuntimeTest, CreateChatWithOnlyATemperatureTakesTheRestFromTheModel) {
  v1::CreateChatRequest request;
  request.mutable_options()->set_max_completion_tokens(8);
  request.mutable_options()->set_temperature(0.5f);
  absl::StatusOr<std::unique_ptr<Chat>> chat = model().CreateChat(request);
  ASSERT_TRUE(chat.ok()) << chat.status();
  const absl::StatusOr<Turn> turn = (*chat)->Send(User("Hi"), nullptr);
  ASSERT_TRUE(turn.ok()) << turn.status();
  EXPECT_FALSE(turn->completion.content().empty())
      << "top_k, top_p and seed are the model's, not zero";
}

// ---- Send

TEST_F(RuntimeTest, AReplyStreamsAndIsWholeAndIsCounted) {
  std::unique_ptr<Chat> chat = CreateChat(64);
  ASSERT_NE(chat, nullptr);
  std::string streamed;
  int pieces = 0;
  const absl::StatusOr<Turn> turn = chat->Send(
      User("Say hello in one short sentence."), [&](const v1::Chunk& chunk) {
        if (chunk.has_content()) streamed += chunk.content();
        ++pieces;
      });
  ASSERT_TRUE(turn.ok()) << turn.status();
  const v1::Completion& reply = turn->completion;
  EXPECT_FALSE(reply.content().empty());
  EXPECT_EQ(reply.content(), streamed) << "the reply is the pieces, joined";
  EXPECT_GT(pieces, 1) << "a reply comes in pieces";
  EXPECT_EQ(reply.finish_reason(), v1::FINISH_REASON_STOP);
  EXPECT_GT(reply.usage().prompt_tokens(), 0);
  EXPECT_GT(reply.usage().completion_tokens(), 0);
  EXPECT_EQ(reply.usage().total_tokens(),
            reply.usage().prompt_tokens() + reply.usage().completion_tokens());
  const v1::ModelDebugger& debugger = reply.model_debugger();
  EXPECT_EQ(debugger.turn(), 1);
  EXPECT_EQ(debugger.context_token_count(), reply.usage().total_tokens());
  EXPECT_FALSE(debugger.prompt_tokens().empty());
  EXPECT_NE(PromptText(debugger).find("Say hello"), std::string::npos)
      << "the prompt tokens spell the prompt";
}

TEST_F(RuntimeTest, AReplyIsCutAtMaxCompletionTokens) {
  std::unique_ptr<Chat> chat = CreateChat(4);
  ASSERT_NE(chat, nullptr);
  const absl::StatusOr<Turn> turn =
      chat->Send(User("Count from one to one hundred."), nullptr);
  ASSERT_TRUE(turn.ok()) << turn.status();
  EXPECT_EQ(turn->completion.finish_reason(), v1::FINISH_REASON_LENGTH);
  EXPECT_LE(turn->completion.usage().completion_tokens(), 4);
}

TEST_F(RuntimeTest, AReplyIsCutWhenTheContextIsFull) {
  // The context the model is loaded with fills before the reply's own limit.
  // On the CPU, LiteRT-LM cannot run this model with a smaller context.
  const int context_length = Request().context_length();
  std::unique_ptr<Chat> chat = CreateChat(4 * context_length);
  ASSERT_NE(chat, nullptr);
  const absl::StatusOr<Turn> turn = chat->Send(
      User("Count from one to one thousand, one number per line."), nullptr);
  ASSERT_TRUE(turn.ok()) << turn.status();
  EXPECT_EQ(turn->completion.finish_reason(), v1::FINISH_REASON_LENGTH);
  EXPECT_LT(turn->completion.usage().completion_tokens(), context_length);
  EXPECT_EQ(turn->completion.usage().total_tokens(), context_length);
}

TEST_F(RuntimeTest, ChineseArrivesInWholeCharacters) {
  std::unique_ptr<Chat> chat = CreateChat(48);
  ASSERT_NE(chat, nullptr);
  const absl::StatusOr<Turn> turn =
      chat->Send(User("用一句中文说你好。"), [&](const v1::Chunk& chunk) {
        const std::string& text =
            chunk.has_content() ? chunk.content() : chunk.reasoning_content();
        // A UTF-8 piece never starts with a continuation byte.
        EXPECT_TRUE(text.empty() ||
                    (static_cast<unsigned char>(text[0]) & 0xC0) != 0x80)
            << text;
      });
  ASSERT_TRUE(turn.ok()) << turn.status();
}

TEST_F(RuntimeTest, TurnsAreNumberedAndTheSecondPromptIsTheTurnAlone) {
  std::unique_ptr<Chat> chat = CreateChat(
      16, {Message(v1::ChatMessage::SYSTEM, "You are a terse assistant."),
           User("My name is Ada."),
           Message(v1::ChatMessage::ASSISTANT, "Hello Ada.")});
  ASSERT_NE(chat, nullptr);
  const absl::StatusOr<Turn> first =
      chat->Send(User("What is my name?"), nullptr);
  ASSERT_TRUE(first.ok()) << first.status();
  const std::string first_prompt =
      PromptText(first->completion.model_debugger());
  EXPECT_NE(first_prompt.find("My name is Ada."), std::string::npos)
      << "before the first turn, the messages so far are still to come";
  EXPECT_NE(first_prompt.find("What is my name?"), std::string::npos);
  EXPECT_EQ(first->completion.model_debugger().turn(), 1);

  const absl::StatusOr<Turn> second = chat->Send(User("And yours?"), nullptr);
  ASSERT_TRUE(second.ok()) << second.status();
  const std::string second_prompt =
      PromptText(second->completion.model_debugger());
  EXPECT_EQ(second_prompt.find("My name is Ada."), std::string::npos)
      << "after it, only the turn itself";
  EXPECT_NE(second_prompt.find("And yours?"), std::string::npos);
  EXPECT_EQ(second->completion.model_debugger().turn(), 2);
  EXPECT_EQ(second->capture_dir, first->capture_dir)
      << "one chat, one capture directory";
}

TEST_F(RuntimeTest, CancelFromTheCallbackEndsTheChat) {
  std::unique_ptr<Chat> chat = CreateChat(512);
  ASSERT_NE(chat, nullptr);
  int pieces = 0;
  const absl::StatusOr<Turn> turn = chat->Send(
      User("Count from one to one thousand."), [&](const v1::Chunk&) {
        if (++pieces == 2) chat->Cancel();
      });
  EXPECT_TRUE(absl::IsCancelled(turn.status())) << turn.status();
  EXPECT_GE(pieces, 2);
  EXPECT_TRUE(
      absl::IsFailedPrecondition(chat->Send(User("Again?"), nullptr).status()))
      << "a cancelled chat is over";
}

TEST_F(RuntimeTest, CancelWithNothingInProgressDoesNothing) {
  std::unique_ptr<Chat> chat = CreateChat(8);
  ASSERT_NE(chat, nullptr);
  chat->Cancel();
  EXPECT_TRUE(chat->Send(User("Hi"), nullptr).ok());
}

TEST_F(RuntimeTest, ReasoningArrivesApartFromTheAnswer) {
  std::unique_ptr<Chat> chat = CreateChat(96, {}, /*thinking=*/true);
  ASSERT_NE(chat, nullptr);
  std::string reasoning;
  std::string content;
  const absl::StatusOr<Turn> turn = chat->Send(
      User("What is 2 + 2? Think first."), [&](const v1::Chunk& chunk) {
        if (chunk.has_reasoning_content())
          reasoning += chunk.reasoning_content();
        if (chunk.has_content()) content += chunk.content();
      });
  ASSERT_TRUE(turn.ok()) << turn.status();
  EXPECT_EQ(turn->completion.reasoning_content(), reasoning);
  EXPECT_EQ(turn->completion.content(), content);
  EXPECT_FALSE(reasoning.empty()) << "the model was asked to think";
}

TEST_F(RuntimeTest, CaptureDirectoryIsUnderTheDataDirectory) {
  std::unique_ptr<Chat> chat = CreateChat(8);
  ASSERT_NE(chat, nullptr);
  const absl::StatusOr<Turn> turn = chat->Send(User("Hi"), nullptr);
  ASSERT_TRUE(turn.ok()) << turn.status();
  EXPECT_EQ(turn->capture_dir.string().rfind(data_dir().string(), 0), 0u)
      << turn->capture_dir;
  EXPECT_TRUE(std::filesystem::is_directory(turn->capture_dir));
}

TEST_F(RuntimeTest, TextThatIsNotUtf8IsRefused) {
  std::unique_ptr<Chat> chat = CreateChat(8);
  ASSERT_NE(chat, nullptr);
  EXPECT_TRUE(
      absl::IsInvalidArgument(chat->Send(User("caf\xE9"), nullptr).status()));
}

TEST_F(RuntimeTest, ATurnThatIsNotAUserMessageIsRefused) {
  std::unique_ptr<Chat> chat = CreateChat(8);
  ASSERT_NE(chat, nullptr);
  EXPECT_TRUE(absl::IsInvalidArgument(
      chat->Send(Message(v1::ChatMessage::ASSISTANT, "I speak first."), nullptr)
          .status()));
}

}  // namespace
}  // namespace model_debugger::runner::llm
