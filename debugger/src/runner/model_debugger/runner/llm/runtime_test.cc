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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "google/protobuf/struct.pb.h"
#include "model_debugger/protocol/runner.pb.h"
#include "model_debugger/runner/llm/litert_lm/litert_lm_runtime.h"
#include "model_debugger/runner/testing/test_utils.h"
#include "model_debugger/runner/util/json.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"

namespace model_debugger::runner::llm {
namespace {

protocol::ChatMessage Message(protocol::ChatMessage::Role role,
                              std::string text) {
  protocol::ChatMessage message;
  message.set_role(role);
  message.set_content(std::move(text));
  return message;
}

protocol::ChatMessage User(std::string text) {
  return Message(protocol::ChatMessage::USER, std::move(text));
}

// Returns the texts of the prompt the turn of `debugger` fed, joined: the
// tokens up to the first decode step, which feeds the prompt's last, without
// the first, the start token or the token held from the turn before.
std::string PromptText(const protocol::ModelDebugger& debugger) {
  std::string text;
  for (int i = 1; i < debugger.fed_tokens_size(); ++i) {
    text += debugger.fed_tokens(i).text();
    if (debugger.fed_tokens(i).phase() == protocol::FedToken::DECODE) break;
  }
  return text;
}

// Moves what the turn wrote to `capture_dir` into `turn_dir`, as the Runner
// does after each turn: every file but the tokens file, which it copies by
// line.
void FileTurn(const std::filesystem::path& capture_dir,
              const std::filesystem::path& turn_dir) {
  for (const auto& entry : std::filesystem::directory_iterator(capture_dir)) {
    const std::string name = entry.path().filename().string();
    if (name != "generated_tokens.jsonl") {
      std::filesystem::rename(entry.path(), turn_dir / name);
    }
  }
}

// A fed row, as (phase, position).
using Row = std::pair<int, int>;

// Returns every row the forward passes captured in `dir` fed, sorted: of each
// <signature>_pre_input_pos_step_<step> file, the leading run of positions,
// the rest being padding.
std::vector<Row> CapturedRows(const std::filesystem::path& dir) {
  constexpr std::string_view kInfix = "_pre_input_pos_step_";
  std::vector<Row> rows;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    const std::string name = entry.path().filename().string();
    const size_t infix = name.find(kInfix);
    if (infix == std::string::npos) continue;
    std::ifstream in(entry.path(), std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
    uint64_t header_size = 0;
    std::memcpy(&header_size, bytes.data(), sizeof(header_size));
    const absl::StatusOr<Json> header =
        ParseJson(std::string_view(bytes).substr(8, header_size));
    EXPECT_TRUE(header.ok()) << name;
    if (!header.ok()) continue;
    const Json& tensor = header->at("pre_input_pos");
    EXPECT_EQ(tensor.at("dtype"), "I32") << name;
    const size_t begin = tensor.at("data_offsets")[0].get<size_t>();
    const size_t end = tensor.at("data_offsets")[1].get<size_t>();
    std::vector<int32_t> positions((end - begin) / sizeof(int32_t));
    std::memcpy(positions.data(), bytes.data() + 8 + header_size + begin,
                end - begin);
    const int phase = name.substr(0, infix) == "decode"
                          ? protocol::FedToken::DECODE
                          : protocol::FedToken::PREFILL;
    for (size_t i = 0; i < positions.size(); ++i) {
      if (i > 0 && positions[i] != positions[0] + static_cast<int32_t>(i)) {
        break;
      }
      rows.emplace_back(phase, positions[i]);
    }
  }
  std::sort(rows.begin(), rows.end());
  return rows;
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
    static auto* const loaded = new std::optional<Loaded>();
    const char* model = std::getenv("RUNNER_MODEL");
    if (model == nullptr) GTEST_SKIP() << "RUNNER_MODEL is not set";
    if (!loaded->has_value()) {
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

  static protocol::LoadRequest Request() {
    protocol::LoadRequest request;
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
  std::unique_ptr<Chat> CreateChat(
      int max_completion_tokens,
      std::vector<protocol::ChatMessage> messages = {}, bool thinking = false) {
    protocol::CreateChatRequest request;
    for (protocol::ChatMessage& message : messages) {
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

TEST_F(RuntimeTest, AModelLoadedOnTheGpuIsWhereTheRuntimeSaysItCanRun) {
  if (DeviceFromEnvironment() != protocol::GPU) GTEST_SKIP() << "on the CPU";
  const absl::Status gpu = runtime().CanRunOnGpu();
  EXPECT_TRUE(gpu.ok()) << gpu;
}

// ---- Info

TEST_F(RuntimeTest, RuntimeInfoNamesTheRuntimeAndItsOptions) {
  const protocol::RuntimeInfo& info = runtime().Info();
  EXPECT_EQ(info.name(), protocol::RuntimeInfo::LITERT_LM);
  EXPECT_EQ(info.version().size(), 40u) << "the LiteRT-LM commit, whole";
  EXPECT_NE(info.load_options_schema().find("cpuThreads"), std::string::npos);
  EXPECT_NE(info.chat_options_schema().find("thinking"), std::string::npos);
}

TEST_F(RuntimeTest, ModelInfoHasTheStopTokens) {
  const protocol::LoadResponse& info = model().Info();
  ASSERT_FALSE(info.stop_tokens().empty());
  for (const protocol::StopToken& stop : info.stop_tokens()) {
    EXPECT_TRUE(stop.has_ids() || !stop.text().empty());
    if (stop.has_ids()) EXPECT_FALSE(stop.ids().ids().empty());
  }
}

// Returns what `info` says of the model's own sampling, for the log.
std::string SamplingText(const protocol::LoadResponse& info) {
  if (!info.has_sampler_defaults()) return "none reported";
  const protocol::SamplerDefaults& defaults = info.sampler_defaults();
  std::string text = absl::StrCat(
      "type ", protocol::SamplerDefaults::Type_Name(defaults.type()));
  if (defaults.has_top_k())
    absl::StrAppend(&text, ", top_k ", defaults.top_k());
  if (defaults.has_top_p())
    absl::StrAppend(&text, ", top_p ", defaults.top_p());
  if (defaults.has_temperature()) {
    absl::StrAppend(&text, ", temperature ", defaults.temperature());
  }
  return text;
}

// The model's own sampling is the model's: the test prints what the runtime
// reports, and checks it only against the protocol's ranges.
TEST_F(RuntimeTest, ModelInfoHasTheModelsOwnSamplingIfItReportsAny) {
  const protocol::LoadResponse& info = model().Info();
  std::cout << "The model's own sampling: " << SamplingText(info) << "\n";
  if (!info.has_sampler_defaults()) return;
  const protocol::SamplerDefaults& defaults = info.sampler_defaults();
  EXPECT_GT(defaults.ByteSizeLong(), 0u) << "an empty one is none, unset";
  if (defaults.has_top_k()) EXPECT_GE(defaults.top_k(), 1);
  if (defaults.has_top_p()) {
    EXPECT_GT(defaults.top_p(), 0);
    EXPECT_LE(defaults.top_p(), 1);
  }
  if (defaults.has_temperature()) EXPECT_GT(defaults.temperature(), 0);
}

// ---- CreateChat

// Returns the counts a refusal for no room carries, a Send's or a
// CreateChat's: the tokens the turn or the start needs, and those the context
// has left; -1 for a count it does not carry.
std::pair<int, int> NoRoomCounts(const absl::Status& status) {
  const auto count = [&status](std::string_view payload) {
    int value = -1;
    const std::optional<absl::Cord> text = status.GetPayload(payload);
    if (!text || !absl::SimpleAtoi(std::string(*text), &value)) return -1;
    return value;
  };
  return {count(kTurnTokensPayload), count(kTokensLeftPayload)};
}

TEST_F(RuntimeTest, CreateChatRefusesASystemMessageTheContextHasNoRoomFor) {
  const int context_length = Request().context_length();
  std::string text;
  for (int i = 0; i < 2 * context_length; ++i) text += "one ";
  protocol::CreateChatRequest request;
  request.mutable_options()->set_max_completion_tokens(8);
  *request.add_messages() = Message(protocol::ChatMessage::SYSTEM, text);
  const absl::Status refused = model().CreateChat(request).status();
  EXPECT_TRUE(absl::IsResourceExhausted(refused)) << refused;
  const std::pair<int, int> counts = NoRoomCounts(refused);
  EXPECT_EQ(counts.second, context_length) << "the whole context is left";
  EXPECT_GT(counts.first, context_length);

  // A system message that fits starts a chat that replies.
  std::unique_ptr<Chat> chat =
      CreateChat(8, {Message(protocol::ChatMessage::SYSTEM, "Be terse.")});
  ASSERT_NE(chat, nullptr);
  const absl::StatusOr<Turn> turn = chat->Send(User("Hi"), nullptr);
  ASSERT_TRUE(turn.ok()) << turn.status();
}

TEST_F(RuntimeTest, CreateChatRefusesASystemMessageThatIsNotFirst) {
  protocol::CreateChatRequest request;
  request.mutable_options()->set_max_completion_tokens(8);
  *request.add_messages() = User("Hi");
  *request.add_messages() = Message(protocol::ChatMessage::SYSTEM, "Be terse.");
  EXPECT_TRUE(absl::IsInvalidArgument(model().CreateChat(request).status()));
}

TEST_F(RuntimeTest, CreateChatRefusesARoleItDoesNotKnow) {
  protocol::CreateChatRequest request;
  request.mutable_options()->set_max_completion_tokens(8);
  *request.add_messages() = User("Hi");
  // 4, the role TOOL was.
  *request.add_messages() =
      Message(static_cast<protocol::ChatMessage::Role>(4), "15C and clear");
  EXPECT_TRUE(absl::IsInvalidArgument(model().CreateChat(request).status()));
}

TEST_F(RuntimeTest, CreateChatRefusesAChatOptionTheRuntimeDoesNotHave) {
  protocol::CreateChatRequest request;
  request.mutable_options()->set_max_completion_tokens(8);
  (*request.mutable_options()
        ->mutable_runtime_options()
        ->mutable_fields())["mood"]
      .set_string_value("cheerful");
  EXPECT_TRUE(absl::IsInvalidArgument(model().CreateChat(request).status()));
}

TEST_F(RuntimeTest, CreateChatWithOnlyATemperatureTakesTheRestFromTheModel) {
  protocol::CreateChatRequest request;
  request.mutable_options()->set_max_completion_tokens(8);
  request.mutable_options()->set_temperature(0.5f);
  absl::StatusOr<std::unique_ptr<Chat>> chat = model().CreateChat(request);
  ASSERT_TRUE(chat.ok()) << chat.status();
  const absl::StatusOr<Turn> turn = (*chat)->Send(User("Hi"), nullptr);
  ASSERT_TRUE(turn.ok()) << turn.status();
  EXPECT_FALSE(turn->completion.content().empty())
      << "top_k, top_p and seed are the model's, not zero";
}

// What the server sends for Top P: every sampling value, top_k the model's
// own.
TEST_F(RuntimeTest, ATopPChatAtTheModelsOwnTopKReplies) {
  const protocol::SamplerDefaults& defaults = model().Info().sampler_defaults();
  if (!defaults.has_top_k()) GTEST_SKIP() << "the model sets no top_k";
  protocol::CreateChatRequest request;
  protocol::ChatOptions& options = *request.mutable_options();
  options.set_max_completion_tokens(8);
  options.set_top_k(defaults.top_k());
  options.set_top_p(0.9f);
  options.set_temperature(0.8f);
  options.set_seed(7);
  absl::StatusOr<std::unique_ptr<Chat>> chat = model().CreateChat(request);
  ASSERT_TRUE(chat.ok()) << chat.status();
  const absl::StatusOr<Turn> turn = (*chat)->Send(User("Hi"), nullptr);
  ASSERT_TRUE(turn.ok()) << turn.status();
  EXPECT_FALSE(turn->completion.content().empty());
}

// ---- Send

TEST_F(RuntimeTest, AReplyStreamsAndIsWholeAndIsCounted) {
  std::unique_ptr<Chat> chat = CreateChat(64);
  ASSERT_NE(chat, nullptr);
  std::string streamed;
  int pieces = 0;
  const absl::StatusOr<Turn> turn =
      chat->Send(User("Say hello in one short sentence."),
                 [&](const protocol::Chunk& chunk) {
                   if (chunk.has_content()) streamed += chunk.content();
                   ++pieces;
                 });
  ASSERT_TRUE(turn.ok()) << turn.status();
  const protocol::Completion& reply = turn->completion;
  EXPECT_FALSE(reply.content().empty());
  EXPECT_EQ(reply.content(), streamed) << "the reply is the pieces, joined";
  EXPECT_GT(pieces, 1) << "a reply comes in pieces";
  EXPECT_EQ(reply.finish_reason(), protocol::FINISH_REASON_STOP);
  EXPECT_GT(reply.usage().prompt_tokens(), 0);
  EXPECT_GT(reply.usage().completion_tokens(), 0);
  EXPECT_EQ(reply.usage().total_tokens(),
            reply.usage().prompt_tokens() + reply.usage().completion_tokens());
  const protocol::ModelDebugger& debugger = reply.model_debugger();
  EXPECT_EQ(debugger.turn(), 1);
  EXPECT_EQ(debugger.context_token_count(), reply.usage().total_tokens())
      << "without reasoning, nothing is rewritten";
  ASSERT_FALSE(debugger.fed_tokens().empty());
  const protocol::FedToken& first = debugger.fed_tokens(0);
  EXPECT_EQ(first.position(), 0);
  EXPECT_EQ(first.phase(), protocol::FedToken::PREFILL);
  if (model().Info().has_start_token()) {
    EXPECT_EQ(first.id(), model().Info().start_token());
  }
  EXPECT_NE(PromptText(debugger).find("Say hello"), std::string::npos)
      << "the fed tokens spell the prompt";
  int decodes = 0;
  for (const protocol::FedToken& token : debugger.fed_tokens()) {
    if (token.phase() != protocol::FedToken::DECODE) continue;
    if (++decodes == 1) {
      EXPECT_EQ(token.position() + 1, reply.usage().prompt_tokens())
          << "the reply is generated from the context up to the first step";
    }
    EXPECT_TRUE(token.has_id()) << token.position();
  }
  EXPECT_EQ(decodes, reply.usage().completion_tokens());
}

TEST_F(RuntimeTest, AReplyIsCutAtMaxCompletionTokens) {
  std::unique_ptr<Chat> chat = CreateChat(4);
  ASSERT_NE(chat, nullptr);
  const absl::StatusOr<Turn> turn =
      chat->Send(User("Count from one to one hundred."), nullptr);
  ASSERT_TRUE(turn.ok()) << turn.status();
  EXPECT_EQ(turn->completion.finish_reason(), protocol::FINISH_REASON_LENGTH);
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
  EXPECT_EQ(turn->completion.finish_reason(), protocol::FINISH_REASON_LENGTH);
  EXPECT_LT(turn->completion.usage().completion_tokens(), context_length);
  EXPECT_EQ(turn->completion.usage().total_tokens(), context_length);
}

TEST_F(RuntimeTest, ATurnWithNoRoomIsRefusedAndTheChatGoesOn) {
  const int context_length = Request().context_length();
  std::unique_ptr<Chat> chat = CreateChat(24);
  ASSERT_NE(chat, nullptr);
  const absl::StatusOr<Turn> first = chat->Send(
      User("My name is Ada. Greet me in one short sentence."), nullptr);
  ASSERT_TRUE(first.ok()) << first.status();
  const int context = first->completion.model_debugger().context_token_count();
  const int left = context_length - context;

  // Refused, whatever the turn's length beyond the context: a turn of `words`
  // repeated words needs a few tokens and then about as many again per word.
  const auto refused = [&](int words) {
    std::string text;
    for (int i = 0; i < words; ++i) text += "one ";
    const absl::StatusOr<Turn> turn = chat->Send(User(text), nullptr);
    EXPECT_TRUE(absl::IsResourceExhausted(turn.status())) << turn.status();
    const std::pair<int, int> counts = NoRoomCounts(turn.status());
    EXPECT_EQ(counts.second, left) << "what the context has left";
    EXPECT_GT(counts.first, counts.second);
    return counts.first;
  };
  const int twice = refused(2 * context_length);
  const int four_times = refused(4 * context_length);
  ASSERT_GT(four_times, twice);
  // Longer than what is left, but not than the whole context.
  const double per_word =
      static_cast<double>(four_times - twice) / (2 * context_length);
  const double fixed = twice - per_word * 2 * context_length;
  const int fits_whole =
      refused(static_cast<int>((context_length - fixed) / per_word) - 2);
  EXPECT_LE(fits_whole, context_length);

  // The chat goes on from the first turn, as if nothing had been sent since.
  const absl::StatusOr<Turn> second =
      chat->Send(User("What is my name? Answer with the name alone."), nullptr);
  ASSERT_TRUE(second.ok()) << second.status();
  const protocol::ModelDebugger& debugger = second->completion.model_debugger();
  EXPECT_EQ(debugger.turn(), 2);
  ASSERT_FALSE(debugger.fed_tokens().empty());
  EXPECT_EQ(debugger.fed_tokens(0).position(), context - 1)
      << "the turn starts with the token the first one left held";
  EXPECT_TRUE(debugger.fed_tokens(0).has_id())
      << "and the chat still knows what that token is";
  EXPECT_NE(second->completion.content().find("Ada"), std::string::npos)
      << second->completion.content();
}

TEST_F(RuntimeTest, ChineseArrivesInWholeCharacters) {
  std::unique_ptr<Chat> chat = CreateChat(48);
  ASSERT_NE(chat, nullptr);
  const absl::StatusOr<Turn> turn =
      chat->Send(User("用一句中文说你好。"), [&](const protocol::Chunk& chunk) {
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
      16, {Message(protocol::ChatMessage::SYSTEM, "You are a terse assistant."),
           User("My name is Ada."),
           Message(protocol::ChatMessage::ASSISTANT, "Hello Ada.")});
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

// Sends two turns on a chat with or without reasoning, and checks each turn's
// fed tokens against the rows its capture shows.
void ExpectTheFedTokensAreTheCapturedRows(Chat& chat,
                                          const protocol::LoadResponse& info) {
  const bool cpu = DeviceFromEnvironment() == protocol::CPU;
  int context = 0;
  int number = 0;
  for (const char* question : {"What is 2 + 2?", "What is 3 + 3?"}) {
    ++number;
    const absl::StatusOr<Turn> turn = chat.Send(User(question), nullptr);
    ASSERT_TRUE(turn.ok()) << turn.status();
    const protocol::Completion& reply = turn->completion;
    const protocol::ModelDebugger& debugger = reply.model_debugger();
    const std::filesystem::path turn_dir = TempDir(
        ::testing::UnitTest::GetInstance()->current_test_info()->name() +
        std::string("_") + std::to_string(number));
    FileTurn(turn->capture_dir, turn_dir);

    std::vector<Row> fed;
    for (const protocol::FedToken& token : debugger.fed_tokens()) {
      fed.emplace_back(token.phase(), token.position());
    }
    std::sort(fed.begin(), fed.end());
    EXPECT_EQ(fed, CapturedRows(turn_dir))
        << "turn " << number << ": one fed token per row of every call";

    const protocol::Usage& usage = reply.usage();
    EXPECT_GT(usage.prompt_tokens(), 0) << "turn " << number;
    EXPECT_EQ(usage.total_tokens(),
              usage.prompt_tokens() + usage.completion_tokens());
    ASSERT_FALSE(debugger.fed_tokens().empty());
    const protocol::FedToken& first = debugger.fed_tokens(0);
    EXPECT_EQ(first.position(), number == 1 ? 0 : context - 1)
        << "turn " << number << " starts with the token held before it";
    if (number == 1 && info.has_start_token()) {
      EXPECT_EQ(first.id(), info.start_token());
    }
    EXPECT_EQ(
        debugger.context_token_count(),
        debugger.fed_tokens(debugger.fed_tokens_size() - 1).position() + 2)
        << "turn " << number << ": the last fed position, and one held";

    // The tokens from the first decode step on, and any rewrite after them.
    int steps = 0;
    int rewritten = 0;
    for (int i = 0; i < debugger.fed_tokens_size(); ++i) {
      const protocol::FedToken& token = debugger.fed_tokens(i);
      if (token.phase() == protocol::FedToken::DECODE) {
        if (steps++ == 0) {
          EXPECT_EQ(token.position() + 1, usage.prompt_tokens());
        }
      } else if (steps > 0) {
        if (rewritten++ == 0) {
          EXPECT_EQ(token.position() + 1, usage.prompt_tokens())
              << "the rewrite starts where the reply did";
        }
      }
      // The prompt and the reply are named by what the runtime reports; the
      // rest by their rows, which on the CPU are equal for equal ids.
      if (cpu || (i > 0 && rewritten == 0)) {
        EXPECT_TRUE(token.has_id())
            << "turn " << number << ", position " << token.position();
      }
      if (!token.has_id()) EXPECT_TRUE(token.text().empty());
    }
    EXPECT_EQ(steps, usage.completion_tokens());
    if (!reply.reasoning_content().empty()) {
      EXPECT_GT(rewritten, 0) << "turn " << number
                              << ": the reply is written again without its "
                                 "reasoning";
    } else {
      EXPECT_EQ(rewritten, 0);
      EXPECT_EQ(debugger.context_token_count(), usage.total_tokens());
    }
    context = debugger.context_token_count();
  }
}

TEST_F(RuntimeTest, TheFedTokensAreTheCapturedRows) {
  std::unique_ptr<Chat> chat = CreateChat(512);
  ASSERT_NE(chat, nullptr);
  ExpectTheFedTokensAreTheCapturedRows(*chat, model().Info());
}

TEST_F(RuntimeTest, TheFedTokensAreTheCapturedRowsWithReasoning) {
  std::unique_ptr<Chat> chat = CreateChat(512, {}, /*thinking=*/true);
  ASSERT_NE(chat, nullptr);
  ExpectTheFedTokensAreTheCapturedRows(*chat, model().Info());
}

TEST_F(RuntimeTest, CancelFromTheCallbackEndsTheChat) {
  std::unique_ptr<Chat> chat = CreateChat(512);
  ASSERT_NE(chat, nullptr);
  int pieces = 0;
  const absl::StatusOr<Turn> turn = chat->Send(
      User("Count from one to one thousand."), [&](const protocol::Chunk&) {
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
      User("What is 2 + 2? Think first."), [&](const protocol::Chunk& chunk) {
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
      chat->Send(Message(protocol::ChatMessage::ASSISTANT, "I speak first."),
                 nullptr)
          .status()));
  // 4, the role TOOL was.
  EXPECT_TRUE(absl::IsInvalidArgument(
      chat->Send(Message(static_cast<protocol::ChatMessage::Role>(4), "15C"),
                 nullptr)
          .status()));
}

}  // namespace
}  // namespace model_debugger::runner::llm
