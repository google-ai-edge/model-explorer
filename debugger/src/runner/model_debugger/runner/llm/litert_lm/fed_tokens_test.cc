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

#include "model_debugger/runner/llm/litert_lm/fed_tokens.h"

#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "model_debugger/protocol/runner.pb.h"
#include "model_debugger/runner/util/json.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace model_debugger::runner::llm {
namespace {

using protocol::FedToken;

// One row a call fed: at `position`, the embedding of `id`, or `id` itself
// where the runtime had no embedding for it.
struct Fed {
  int position;
  int32_t id;
  bool raw = false;
};

// The bytes of a row of `floats` floats.
std::string Floats(const std::vector<float>& floats) {
  std::string bytes(floats.size() * sizeof(float), '\0');
  std::memcpy(bytes.data(), floats.data(), bytes.size());
  return bytes;
}

// The embeddings row the runtime looks `row`'s id up as: four floats that
// differ by id, or the id and zeros.
std::string Embedding(const Fed& row) {
  if (row.raw) {
    std::string bytes(4 * sizeof(float), '\0');
    std::memcpy(bytes.data(), &row.id, sizeof(row.id));
    return bytes;
  }
  const float id = static_cast<float>(row.id);
  return Floats({id, id / 2, 1, -id});
}

std::string PerLayerEmbedding(const Fed& row) {
  return Floats({static_cast<float>(row.id) + 0.25f, 3});
}

// A turn's capture, written as the RuntimeDebugger writes one, in a
// directory of its own.
class Capture {
 public:
  explicit Capture(std::string_view name)
      : dir_(std::filesystem::path(::testing::TempDir()) / name) {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
    std::filesystem::create_directories(dir_, ec);
  }

  const std::filesystem::path& dir() const { return dir_; }
  const std::set<std::string>& names() const { return names_; }

  // Writes a call of `signature` at `step` that fed `rows`, its inputs padded
  // to `size` rows as a prefill's are.
  void Call(const std::string& signature, int step,
            const std::vector<Fed>& rows, int size = 1) {
    std::vector<int32_t> positions(size, 0);
    std::string embeddings;
    std::string per_layer;
    for (int i = 0; i < size; ++i) {
      const Fed pad = {0, 0};
      const Fed& row = i < static_cast<int>(rows.size()) ? rows[i] : pad;
      positions[i] = row.position;
      embeddings += i < static_cast<int>(rows.size()) ? Embedding(row)
                                                      : Floats({0, 0, 0, 0});
      per_layer += PerLayerEmbedding(row);
    }
    std::string bytes(positions.size() * sizeof(int32_t), '\0');
    std::memcpy(bytes.data(), positions.data(), bytes.size());
    Write(signature, step, "input_pos", "I32", {size}, bytes);
    Write(signature, step, "embeddings", "F32", {1, size, 4}, embeddings);
    Write(signature, step, "per_layer_embeddings", "F32", {1, size, 1, 2},
          per_layer);
  }

  // Appends to generated_tokens.jsonl a line for each of `ids`.
  void Release(const std::vector<int32_t>& ids) {
    std::ofstream out(dir_ / "generated_tokens.jsonl", std::ios::app);
    for (const int32_t id : ids) {
      Json record = Json::object();
      record["token_ids"] = Json::array({Json::array({id})});
      record["texts"] = Json::array({"x"});
      const absl::StatusOr<std::string> dumped = Dump(record);
      EXPECT_TRUE(dumped.ok());
      if (dumped.ok()) {
        out << *dumped << '\n';
      }
    }
  }

  // Writes `name` with `contents`, which need not be a tensor.
  void WriteFile(const std::string& name, const std::string& contents) {
    std::ofstream(dir_ / name, std::ios::binary) << contents;
    names_.insert(name);
  }

 private:
  void Write(const std::string& signature, int step, const std::string& input,
             const std::string& dtype, const std::vector<int>& shape,
             const std::string& data) {
    const std::string tensor = "pre_" + input;
    Json header = {{tensor,
                    {{"dtype", dtype},
                     {"shape", shape},
                     {"data_offsets", {0, data.size()}}}},
                   {"__metadata__", {{"signature", signature}}}};
    const std::string text = *Dump(header);
    const uint64_t size = text.size();
    std::string file(sizeof(size), '\0');
    std::memcpy(file.data(), &size, sizeof(size));
    WriteFile(signature + "_" + tensor + "_step_" + std::to_string(step) +
                  ".safetensors",
              file + text + data);
  }

  std::filesystem::path dir_;
  std::set<std::string> names_;
};

// One fed token as the test expects it: unset where id is.
struct Expected {
  int position;
  FedToken::Phase phase;
  std::optional<int32_t> id;
};

void ExpectTokens(const FedTurn& turn, const std::vector<Expected>& expected) {
  EXPECT_EQ(turn.tokens.size(), expected.size());
  if (turn.tokens.size() != expected.size()) {
    return;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    const FedToken& token = turn.tokens[i];
    EXPECT_EQ(token.position(), expected[i].position) << "token " << i;
    EXPECT_EQ(token.phase(), expected[i].phase) << "token " << i;
    EXPECT_EQ(token.has_id(), expected[i].id.has_value()) << "token " << i;
    if (token.has_id() && expected[i].id) {
      EXPECT_EQ(token.id(), *expected[i].id) << "token " << i;
    }
  }
}

constexpr FedToken::Phase kPrefill = FedToken::PREFILL;
constexpr FedToken::Phase kDecode = FedToken::DECODE;
constexpr int32_t kStart = 2;
constexpr int32_t kStop = 1;

// The first turn of a chat: the start token and the prompt 10, 11, 12, then
// a reply that sampled 20, 21 and a stop token it did not release. The tokens
// file starts with the lines `earlier`, as if a turn before had released
// them.
Capture FirstTurn(std::string_view name,
                  const std::vector<int32_t>& earlier = {}) {
  Capture capture(name);
  capture.Release(earlier);
  capture.Call("prefill_8", 4, {{0, kStart}, {1, 10}, {2, 11}}, 8);
  capture.Call("decode", 4, {{3, 12}});
  capture.Call("decode", 5, {{4, 20}});
  capture.Call("decode", 6, {{5, 21}});
  capture.Release({20, 21});
  return capture;
}

TurnFacts FirstFacts() {
  TurnFacts facts;
  facts.start_token = kStart;
  facts.prompt = {10, 11, 12};
  facts.context_before = 0;
  facts.context_after = 7;
  facts.decode_steps = 3;
  facts.stop_tokens = {kStop, 50};
  return facts;
}

// The turn after FirstTurn, which held what it sampled last: it feeds that
// token, as `held` (its id, without an embedding), then the prompt 10, 13,
// and samples 22 and a stop token.
Capture SecondTurn(std::string_view name, int32_t held) {
  Capture capture(name);
  capture.Call("prefill_8", 9, {{6, held, /*raw=*/true}, {7, 10}}, 8);
  capture.Call("decode", 9, {{8, 13}});
  capture.Call("decode", 10, {{9, 22}});
  capture.Release({22});
  return capture;
}

// The facts of SecondTurn, after a first turn that left `held`.
TurnFacts SecondFacts(std::optional<std::set<int32_t>> held) {
  TurnFacts facts;
  facts.start_token = kStart;
  facts.prompt = {10, 13};
  facts.context_before = 7;
  facts.context_after = 11;
  facts.decode_steps = 2;
  facts.stop_tokens = {kStop, 50};
  facts.held = std::move(held);
  return facts;
}

TEST(FedTokensTest, TheFirstTurnFeedsTheStartTokenThePromptAndTheReply) {
  const Capture capture = FirstTurn("first");
  TokenRows rows;
  const absl::StatusOr<FedTurn> turn =
      ReadFedTokens(capture.dir(), capture.names(), 0, FirstFacts(), rows);
  ASSERT_TRUE(turn.ok()) << turn.status();
  EXPECT_EQ(turn->first_decode_position, 3);
  ExpectTokens(*turn, {{0, kPrefill, kStart},
                       {1, kPrefill, 10},
                       {2, kPrefill, 11},
                       {3, kDecode, 12},
                       {4, kDecode, 20},
                       {5, kDecode, 21}});
}

TEST(FedTokensTest, OnlyTheTurnsLinesOfTheTokensFileCount) {
  const Capture capture = FirstTurn("lines", {7, 8, 9});
  TokenRows rows;
  const absl::StatusOr<FedTurn> turn =
      ReadFedTokens(capture.dir(), capture.names(), 3, FirstFacts(), rows);
  ASSERT_TRUE(turn.ok()) << turn.status();
  ASSERT_EQ(turn->tokens.size(), 6u);
  EXPECT_EQ(turn->tokens[4].id(), 20);
  EXPECT_EQ(turn->tokens[5].id(), 21);
}

TEST(FedTokensTest, ALaterTurnStartsWithTheHeldTokenItsRowHolds) {
  TokenRows rows;
  const Capture first = FirstTurn("held_first");
  const absl::StatusOr<FedTurn> before =
      ReadFedTokens(first.dir(), first.names(), 0, FirstFacts(), rows);
  ASSERT_TRUE(before.ok()) << before.status();
  // It stopped: what it sampled last is a stop token it did not release.
  EXPECT_EQ(before->held, (std::set<int32_t>{kStop, 50}));

  // The stop token sampled last is held, without an embedding: its row is
  // its id. The prompt 10, 13 follows.
  const Capture second = SecondTurn("held_second", kStop);
  const absl::StatusOr<FedTurn> turn = ReadFedTokens(
      second.dir(), second.names(), 0, SecondFacts(before->held), rows);
  ASSERT_TRUE(turn.ok()) << turn.status();
  EXPECT_EQ(turn->first_decode_position, 8);
  ExpectTokens(*turn, {{6, kPrefill, kStop},
                       {7, kPrefill, 10},
                       {8, kDecode, 13},
                       {9, kDecode, 22}});
}

TEST(FedTokensTest, AHeldTokenTheTurnBeforeDidNotSampleIsLeftUnset) {
  // The row holds 7, which is no stop token: the turn before cannot have
  // stopped on it.
  const Capture capture = SecondTurn("held_other", 7);
  TokenRows rows;
  const absl::StatusOr<FedTurn> turn =
      ReadFedTokens(capture.dir(), capture.names(), 0,
                    SecondFacts(std::set<int32_t>{kStop, 50}), rows);
  ASSERT_TRUE(turn.ok()) << turn.status();
  ExpectTokens(*turn, {{6, kPrefill, std::nullopt},
                       {7, kPrefill, 10},
                       {8, kDecode, 13},
                       {9, kDecode, 22}});
}

TEST(FedTokensTest, AHeldRowWhoseBytesNameAnotherIdIsNamedNowhereInTheTurn) {
  TokenRows rows;
  // The first turn's reply teaches the row of 20, and it stopped.
  const Capture first = FirstTurn("held_keyed_first");
  const absl::StatusOr<FedTurn> before =
      ReadFedTokens(first.dir(), first.names(), 0, FirstFacts(), rows);
  ASSERT_TRUE(before.ok()) << before.status();

  // The second turn starts with that row as an embedding, though the turn
  // before left a stop token held; its reply then feeds the same row as 20.
  Capture second("held_keyed_second");
  second.Call("prefill_8", 9, {{6, 20}, {7, 10}}, 8);
  second.Call("decode", 9, {{8, 13}});
  second.Call("decode", 10, {{9, 20}});
  second.Call("prefill_8", 10, {{8, 13, /*raw=*/true}, {9, 20}}, 8);
  second.Release({20});
  const absl::StatusOr<FedTurn> turn = ReadFedTokens(
      second.dir(), second.names(), 0, SecondFacts(before->held), rows);
  ASSERT_TRUE(turn.ok()) << turn.status();
  // Its bytes are named nowhere, not by the reply nor by the rewrite.
  ExpectTokens(*turn, {{6, kPrefill, std::nullopt},
                       {7, kPrefill, 10},
                       {8, kDecode, 13},
                       {9, kDecode, std::nullopt},
                       {8, kPrefill, 13},
                       {9, kPrefill, std::nullopt}});
}

TEST(FedTokensTest, AReplyCutShortLeavesHeldTheLastTokenItReleased) {
  // The reply released all it sampled, 20, 21 and 23: it was cut short.
  Capture first("cut_first");
  first.Call("prefill_8", 4, {{0, kStart}, {1, 10}, {2, 11}}, 8);
  first.Call("decode", 4, {{3, 12}});
  first.Call("decode", 5, {{4, 20}});
  first.Call("decode", 6, {{5, 21}});
  first.Release({20, 21, 23});
  TokenRows rows;
  const absl::StatusOr<FedTurn> before =
      ReadFedTokens(first.dir(), first.names(), 0, FirstFacts(), rows);
  ASSERT_TRUE(before.ok()) << before.status();
  EXPECT_EQ(before->held, std::set<int32_t>{23});

  const Capture held = SecondTurn("cut_held", 23);
  const absl::StatusOr<FedTurn> named = ReadFedTokens(
      held.dir(), held.names(), 0, SecondFacts(before->held), rows);
  ASSERT_TRUE(named.ok()) << named.status();
  EXPECT_EQ(named->tokens[0].id(), 23);
  // A stop token is not what it sampled last.
  const Capture stop = SecondTurn("cut_stop", kStop);
  const absl::StatusOr<FedTurn> unset = ReadFedTokens(
      stop.dir(), stop.names(), 0, SecondFacts(before->held), rows);
  ASSERT_TRUE(unset.ok()) << unset.status();
  EXPECT_FALSE(unset->tokens[0].has_id());
}

// A turn with reasoning: the reply sampled 30, 31 (reasoning), 20, 21 and a
// stop token; the runtime then rewrote the context from position 3 with the
// held 12, without its embedding, and the answer 20 and `last`.
Capture ThinkingTurn(std::string_view name, int32_t last) {
  Capture capture(name);
  capture.Call("prefill_8", 4, {{0, kStart}, {1, 10}, {2, 11}}, 8);
  capture.Call("decode", 4, {{3, 12}});
  capture.Call("decode", 5, {{4, 30}});
  capture.Call("decode", 6, {{5, 31}});
  capture.Call("decode", 7, {{6, 20}});
  capture.Call("decode", 8, {{7, 21}});
  capture.Call("prefill_8", 7, {{3, 12, /*raw=*/true}, {4, 20}, {5, last}}, 8);
  capture.Release({30, 31, 20, 21});
  return capture;
}

TurnFacts ThinkingFacts() {
  TurnFacts facts = FirstFacts();
  facts.context_after = 7;
  facts.decode_steps = 5;
  return facts;
}

TEST(FedTokensTest, ARewriteAfterTheReplyIsNamedByItsRows) {
  const Capture capture = ThinkingTurn("thinking", 21);
  TokenRows rows;
  const absl::StatusOr<FedTurn> turn =
      ReadFedTokens(capture.dir(), capture.names(), 0, ThinkingFacts(), rows);
  ASSERT_TRUE(turn.ok()) << turn.status();
  EXPECT_EQ(turn->first_decode_position, 3);
  // What it leaves held only its row will name.
  EXPECT_FALSE(turn->held.has_value());
  ExpectTokens(*turn, {{0, kPrefill, kStart},
                       {1, kPrefill, 10},
                       {2, kPrefill, 11},
                       {3, kDecode, 12},
                       {4, kDecode, 30},
                       {5, kDecode, 31},
                       {6, kDecode, 20},
                       {7, kDecode, 21},
                       {3, kPrefill, 12},
                       {4, kPrefill, 20},
                       {5, kPrefill, 21}});
}

TEST(FedTokensTest, ARewriteRowSeenNowhereIsLeftUnset) {
  const Capture capture = ThinkingTurn("unseen", 99);
  TokenRows rows;
  const absl::StatusOr<FedTurn> turn =
      ReadFedTokens(capture.dir(), capture.names(), 0, ThinkingFacts(), rows);
  ASSERT_TRUE(turn.ok()) << turn.status();
  ASSERT_EQ(turn->tokens.size(), 11u);
  EXPECT_EQ(turn->tokens[9].id(), 20);
  EXPECT_FALSE(turn->tokens[10].has_id());
}

TEST(FedTokensTest, APromptOfAnotherLengthLeavesThePromptUnset) {
  const Capture capture = FirstTurn("length");
  TurnFacts facts = FirstFacts();
  facts.prompt = {10, 11, 12, 13};
  TokenRows rows;
  const absl::StatusOr<FedTurn> turn =
      ReadFedTokens(capture.dir(), capture.names(), 0, facts, rows);
  ASSERT_TRUE(turn.ok()) << turn.status();
  ExpectTokens(*turn, {{0, kPrefill, std::nullopt},
                       {1, kPrefill, std::nullopt},
                       {2, kPrefill, std::nullopt},
                       {3, kDecode, std::nullopt},
                       {4, kDecode, 20},
                       {5, kDecode, 21}});
}

TEST(FedTokensTest, RowsThatContradictTheirIdsLeaveTheirGroupUnset) {
  // Positions 1 and 2 fed the same row, which the prompt names twice over.
  Capture capture("contradiction");
  capture.Call("prefill_8", 4, {{0, kStart}, {1, 10}, {2, 10}}, 8);
  capture.Call("decode", 4, {{3, 12}});
  capture.Call("decode", 5, {{4, 20}});
  capture.Call("decode", 6, {{5, 21}});
  capture.Release({20, 21});
  TokenRows rows;
  const absl::StatusOr<FedTurn> turn =
      ReadFedTokens(capture.dir(), capture.names(), 0, FirstFacts(), rows);
  ASSERT_TRUE(turn.ok()) << turn.status();
  ASSERT_EQ(turn->tokens.size(), 6u);
  for (int i = 0; i < 4; ++i) EXPECT_FALSE(turn->tokens[i].has_id()) << i;
  EXPECT_EQ(turn->tokens[4].id(), 20) << "the reply is a group of its own";
}

TEST(FedTokensTest, APromptAndAReplyThatDisagreeLeaveBothUnset) {
  // The prompt was rendered 10, 11, 12, but position 2 fed the row the reply
  // feeds as 20: one of the two is wrong, and which cannot be told.
  Capture capture("disagree");
  capture.Call("prefill_8", 4, {{0, kStart}, {1, 10}, {2, 20}}, 8);
  capture.Call("decode", 4, {{3, 12}});
  capture.Call("decode", 5, {{4, 20}});
  capture.Call("decode", 6, {{5, 21}});
  capture.Call("decode", 7, {{6, 30}});
  capture.Call("prefill_8", 6, {{3, 12, /*raw=*/true}, {4, 20}, {5, 21}}, 8);
  capture.Release({20, 21, 30});
  TurnFacts facts = FirstFacts();
  facts.context_after = 7;
  facts.decode_steps = 4;
  TokenRows rows;
  const absl::StatusOr<FedTurn> turn =
      ReadFedTokens(capture.dir(), capture.names(), 0, facts, rows);
  ASSERT_TRUE(turn.ok()) << turn.status();
  // Neither group names a row of the rewrite either: what they would have
  // taught is not learned.
  ExpectTokens(*turn, {{0, kPrefill, std::nullopt},
                       {1, kPrefill, std::nullopt},
                       {2, kPrefill, std::nullopt},
                       {3, kDecode, std::nullopt},
                       {4, kDecode, std::nullopt},
                       {5, kDecode, std::nullopt},
                       {6, kDecode, std::nullopt},
                       {3, kPrefill, 12},
                       {4, kPrefill, std::nullopt},
                       {5, kPrefill, std::nullopt}});
}

TEST(FedTokensTest, ARowContradictedLaterIsForgottenForTheRestOfTheChat) {
  TokenRows rows;
  // The first turn's reply teaches the row of 20.
  const Capture first = FirstTurn("forgotten_first");
  ASSERT_TRUE(
      ReadFedTokens(first.dir(), first.names(), 0, FirstFacts(), rows).ok());

  // The second turn's prompt, rendered 13, fed that row; its reasoning is
  // then rewritten, the row of 20 among the rewrite's.
  Capture second("forgotten_second");
  second.Call("prefill_8", 9, {{6, kStop, /*raw=*/true}, {7, 20}}, 8);
  second.Call("decode", 9, {{8, 14}});
  second.Call("decode", 10, {{9, 22}});
  second.Call("prefill_8", 10, {{8, 14, /*raw=*/true}, {9, 20}}, 8);
  second.Release({22});
  TurnFacts facts = SecondFacts(std::nullopt);
  facts.prompt = {13, 14};
  const absl::StatusOr<FedTurn> turn =
      ReadFedTokens(second.dir(), second.names(), 0, facts, rows);
  ASSERT_TRUE(turn.ok()) << turn.status();
  ExpectTokens(*turn, {{6, kPrefill, std::nullopt},
                       {7, kPrefill, std::nullopt},
                       {8, kDecode, std::nullopt},
                       {9, kDecode, 22},
                       {8, kPrefill, 14},
                       {9, kPrefill, std::nullopt}});
}

TEST(FedTokensTest, ADecodeCounterThatDisagreesLeavesEveryIdUnset) {
  const Capture capture = FirstTurn("counter");
  TurnFacts facts = FirstFacts();
  facts.decode_steps = 4;
  TokenRows rows;
  const absl::StatusOr<FedTurn> turn =
      ReadFedTokens(capture.dir(), capture.names(), 0, facts, rows);
  ASSERT_TRUE(turn.ok()) << turn.status();
  ExpectTokens(*turn, {{0, kPrefill, std::nullopt},
                       {1, kPrefill, std::nullopt},
                       {2, kPrefill, std::nullopt},
                       {3, kDecode, std::nullopt},
                       {4, kDecode, std::nullopt},
                       {5, kDecode, std::nullopt}});
  EXPECT_EQ(turn->first_decode_position, 3) << "positions are still measured";
  EXPECT_FALSE(turn->held.has_value());
}

TEST(FedTokensTest, AFileThatCannotBeReadIsInternal) {
  Capture capture = FirstTurn("unreadable");
  capture.WriteFile("decode_pre_input_pos_step_9.safetensors", "not a tensor");
  TokenRows rows;
  EXPECT_TRUE(absl::IsInternal(
      ReadFedTokens(capture.dir(), capture.names(), 0, FirstFacts(), rows)
          .status()));
}

TEST(FedTokensTest, ATurnWithoutADecodeStepIsInternal) {
  Capture capture("no_decode");
  capture.Call("prefill_8", 4, {{0, kStart}, {1, 10}}, 8);
  TokenRows rows;
  EXPECT_TRUE(absl::IsInternal(
      ReadFedTokens(capture.dir(), capture.names(), 0, FirstFacts(), rows)
          .status()));
}

}  // namespace
}  // namespace model_debugger::runner::llm
