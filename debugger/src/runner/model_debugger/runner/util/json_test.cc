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

#include "model_debugger/runner/util/json.h"

#include <string>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace model_debugger::runner {
namespace {

TEST(JsonTest, ParsesOrSaysItIsNotJson) {
  const absl::StatusOr<Json> message =
      ParseJson(R"({"role": "assistant", "content": "Hi"})");
  ASSERT_TRUE(message.ok()) << message.status();
  EXPECT_EQ(*message, (Json{{"role", "assistant"}, {"content", "Hi"}}));

  EXPECT_TRUE(absl::IsInvalidArgument(ParseJson("{not json").status()));
  EXPECT_TRUE(absl::IsInvalidArgument(ParseJson("").status()));
  EXPECT_FALSE(ParseJson("{not json").status().message().empty());
}

// Checks that a prompt reaches the runtime exactly as it was given.
TEST(JsonTest, AStringSurvivesBeingWrittenAndRead) {
  const std::string prompt =
      "say \"hi\"\\n\n\tint main() { return '\\0'; }\x01 caf\xC3\xA9 "
      "\xF0\x9F\x99\x82";
  const Json message = {
      {"role", "user"},
      {"content", Json::array({{{"type", "text"}, {"text", prompt}}})}};

  const absl::StatusOr<std::string> text = Dump(message);
  ASSERT_TRUE(text.ok()) << text.status();
  EXPECT_EQ(text->find('\n'), std::string::npos)
      << "a line end is escaped, not written";
  const absl::StatusOr<Json> read = ParseJson(*text);
  ASSERT_TRUE(read.ok()) << read.status();
  EXPECT_EQ(*read, message);
}

TEST(JsonTest, BytesThatAreNotUtf8AreRefusedReadOrWritten) {
  EXPECT_TRUE(absl::IsInvalidArgument(
      ParseJson("{\"content\": \"caf\xE9\"}").status()));

  EXPECT_TRUE(absl::IsInvalidArgument(
      Dump(Json{{"content", std::string("caf\xE9")}}).status()));
  EXPECT_TRUE(absl::IsInvalidArgument(
      Dump(Json{{"content", std::string("\xE4\xBD")}}).status()))
      << "two of the three bytes of a character";
  EXPECT_TRUE(absl::IsInvalidArgument(
      Dump(Json::array({{{"text", std::string("\xFF")}}})).status()))
      << "inside an array";
}

TEST(JsonTest, AReplacementCharacterThatWasGivenIsWritten) {
  const absl::StatusOr<std::string> text =
      Dump(Json{{"content", std::string("caf\xEF\xBF\xBD")}});
  ASSERT_TRUE(text.ok()) << text.status();
  EXPECT_EQ(*text, "{\"content\":\"caf\xEF\xBF\xBD\"}");
}

}  // namespace
}  // namespace model_debugger::runner
