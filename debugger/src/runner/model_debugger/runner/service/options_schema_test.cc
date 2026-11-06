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

#include "model_debugger/runner/service/options_schema.h"

#include <cmath>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>

#include "google/protobuf/struct.pb.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"

namespace model_debugger::runner::service {
namespace {

using google::protobuf::Struct;
using google::protobuf::Value;

// A schema with every keyword the validator understands.
constexpr std::string_view kSchema = R"({
  "type": "object",
  "properties": {
    "threads": {"type": "integer", "minimum": 1, "maximum": 8},
    "scale": {"type": "number", "minimum": 0},
    "fast": {"type": "boolean"},
    "mode": {"enum": ["a", "b"]},
    "name": {"type": ["string", "null"]}
  },
  "required": ["mode"],
  "additionalProperties": false
})";

constexpr std::string_view kField = "LoadRequest.runtime_options";

Struct Options(std::initializer_list<std::pair<const char*, Value>> fields) {
  Struct options;
  for (const auto& [name, value] : fields) {
    (*options.mutable_fields())[name] = value;
  }
  return options;
}

Value Number(double number) {
  Value value;
  value.set_number_value(number);
  return value;
}

Value Text(const char* text) {
  Value value;
  value.set_string_value(text);
  return value;
}

Value Bool(bool flag) {
  Value value;
  value.set_bool_value(flag);
  return value;
}

Value Null() {
  Value value;
  value.set_null_value(google::protobuf::NULL_VALUE);
  return value;
}

// The status of `options` against kSchema.
absl::Status Validate(const Struct& options) {
  return ValidateOptions(kSchema, options, kField);
}

TEST(ValidateOptionsTest, AcceptsWhatTheSchemaAllows) {
  EXPECT_TRUE(Validate(Options({{"mode", Text("a")},
                                {"threads", Number(8)},
                                {"scale", Number(0.5)},
                                {"fast", Bool(true)},
                                {"name", Null()}}))
                  .ok());
  EXPECT_TRUE(
      Validate(Options({{"mode", Text("b")}, {"threads", Number(1)}})).ok())
      << "the bounds are inclusive";
  EXPECT_TRUE(
      Validate(Options({{"mode", Text("b")}, {"threads", Number(2.0)}})).ok())
      << "2.0 is an integer, as JSON Schema counts";
}

TEST(ValidateOptionsTest, RefusesAValueOfTheWrongType) {
  const absl::Status status =
      Validate(Options({{"mode", Text("a")}, {"fast", Text("yes")}}));
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_EQ(status.message(),
            "LoadRequest.runtime_options.fast is \"yes\"; it must be of type "
            "\"boolean\".");
  EXPECT_TRUE(absl::IsInvalidArgument(
      Validate(Options({{"mode", Text("a")}, {"threads", Number(2.5)}}))));
  EXPECT_TRUE(absl::IsInvalidArgument(
      Validate(Options({{"mode", Text("a")}, {"name", Number(1)}}))))
      << "neither of the types listed";
}

TEST(ValidateOptionsTest, RefusesANumberOutOfRange) {
  absl::Status status =
      Validate(Options({{"mode", Text("a")}, {"threads", Number(0)}}));
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_EQ(status.message(),
            "LoadRequest.runtime_options.threads is 0.0; it must be at least "
            "1.");
  status = Validate(Options({{"mode", Text("a")}, {"threads", Number(9)}}));
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_EQ(status.message(),
            "LoadRequest.runtime_options.threads is 9.0; it must be at most "
            "8.");
}

TEST(ValidateOptionsTest, RefusesANumberThatIsNotFinite) {
  for (const double number : {std::nan(""), HUGE_VAL, -HUGE_VAL}) {
    const absl::Status status =
        Validate(Options({{"mode", Text("a")}, {"scale", Number(number)}}));
    EXPECT_TRUE(absl::IsInvalidArgument(status)) << number;
    EXPECT_EQ(status.message(),
              "LoadRequest.runtime_options.scale is not a finite number.");
  }
}

TEST(ValidateOptionsTest, RefusesANumberThatIsNotFiniteWhereverItIs) {
  constexpr std::string_view kObject = R"({"type": "object"})";
  absl::Status status =
      ValidateOptions(kObject, Options({{"x", Number(std::nan(""))}}), kField);
  EXPECT_TRUE(absl::IsInvalidArgument(status))
      << "in a member the schema does not list: " << status;
  EXPECT_EQ(status.message(),
            "LoadRequest.runtime_options.x is not a finite number.");

  Value list;
  list.mutable_list_value()->add_values()->set_number_value(1);
  list.mutable_list_value()->add_values()->set_number_value(HUGE_VAL);
  status = ValidateOptions(kObject, Options({{"x", list}}), kField);
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << "in a list: " << status;
  EXPECT_EQ(status.message(),
            "LoadRequest.runtime_options.x[1] is not a finite number.");
}

TEST(ValidateOptionsTest, RefusesAValueNotListed) {
  const absl::Status status = Validate(Options({{"mode", Text("c")}}));
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_EQ(status.message(),
            "LoadRequest.runtime_options.mode is \"c\"; it must be one of "
            "[\"a\",\"b\"].");
}

TEST(ValidateOptionsTest, RefusesAnOptionTheSchemaDoesNotHave) {
  const absl::Status status =
      Validate(Options({{"mode", Text("a")}, {"turbo", Bool(true)}}));
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_EQ(status.message(),
            "LoadRequest.runtime_options has turbo, which is not an option of "
            "this runtime.");
}

TEST(ValidateOptionsTest, RefusesAMissingRequiredOption) {
  const absl::Status status = Validate(Options({{"fast", Bool(true)}}));
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_EQ(status.message(),
            "LoadRequest.runtime_options lacks mode, which is required.");
}

TEST(ValidateOptionsTest, AKeywordNotUnderstoodFailsEveryCall) {
  constexpr std::string_view kPattern = R"({
    "type": "object",
    "properties": {"name": {"type": "string", "pattern": "^[a-z]+$"}}
  })";
  const absl::Status status = ValidateOptions(kPattern, Struct(), kField);
  EXPECT_TRUE(absl::IsInternal(status))
      << "even with no option given: " << status;
  EXPECT_NE(status.message().find("#/properties/name/pattern"),
            std::string_view::npos)
      << status;
}

TEST(ValidateOptionsTest, ASchemaThatIsNotJsonFailsEveryCall) {
  EXPECT_TRUE(absl::IsInternal(ValidateOptions("{", Struct(), kField)));
  EXPECT_TRUE(absl::IsInternal(ValidateOptions("", Struct(), kField)));
}

}  // namespace
}  // namespace model_debugger::runner::service
