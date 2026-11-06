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
#include <cstddef>
#include <string>
#include <string_view>

#include "google/protobuf/struct.pb.h"
#include "model_debugger/runner/util/json.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace model_debugger::runner::service {
namespace {

using google::protobuf::Struct;
using google::protobuf::Value;

// The types the keyword "type" may name.
constexpr std::string_view kTypes[] = {"null",   "boolean", "object", "array",
                                       "number", "string",  "integer"};

Json ToJson(const Struct& object);

Json ToJson(const Value& value) {
  switch (value.kind_case()) {
    case Value::kBoolValue:
      return value.bool_value();
    case Value::kNumberValue:
      return value.number_value();
    case Value::kStringValue:
      return value.string_value();
    case Value::kStructValue:
      return ToJson(value.struct_value());
    case Value::kListValue: {
      Json array = Json::array();
      for (const Value& item : value.list_value().values()) {
        array.push_back(ToJson(item));
      }
      return array;
    }
    default:  // A null, or a value of no kind, which JSON writes as null.
      return nullptr;
  }
}

Json ToJson(const Struct& object) {
  Json json = Json::object();
  for (const auto& field : object.fields()) {
    json[field.first] = ToJson(field.second);
  }
  return json;
}

// `value` as JSON text, for a message: bytes that are not UTF-8 are replaced,
// since the dump that refuses them does so by throwing.
std::string Show(const Json& value) {
  return value.dump(/*indent=*/-1, /*indent_char=*/' ', /*ensure_ascii=*/false,
                    Json::error_handler_t::replace);
}

absl::Status NotUnderstood(std::string_view where, std::string_view what) {
  return absl::InternalError(
      absl::StrCat("The runtime's options schema ", what, " at ", where,
                   ", which the runner does not check."));
}

bool IsTypeName(const Json& name) {
  if (!name.is_string()) return false;
  for (const std::string_view type : kTypes) {
    if (name.get_ref<const std::string&>() == type) return true;
  }
  return false;
}

// Returns Internal if `schema`, found at `where`, uses a keyword or a form of
// one that Check does not understand, there or in a schema inside it.
absl::Status CheckKeywords(const Json& schema, const std::string& where) {
  if (!schema.is_object()) return NotUnderstood(where, "is not an object");
  for (const auto& item : schema.items()) {
    const std::string& keyword = item.key();
    const Json& value = item.value();
    const std::string at = absl::StrCat(where, "/", keyword);
    if (keyword == "type") {
      bool understood = IsTypeName(value);
      if (value.is_array()) {
        understood = true;
        for (const Json& name : value)
          understood = understood && IsTypeName(name);
      }
      if (!understood) return NotUnderstood(at, "names no type it knows");
    } else if (keyword == "properties") {
      if (!value.is_object()) return NotUnderstood(at, "is not an object");
      for (const auto& property : value.items()) {
        if (absl::Status status = CheckKeywords(
                property.value(), absl::StrCat(at, "/", property.key()));
            !status.ok()) {
          return status;
        }
      }
    } else if (keyword == "required") {
      bool understood = value.is_array();
      if (understood) {
        for (const Json& name : value)
          understood = understood && name.is_string();
      }
      if (!understood) return NotUnderstood(at, "is not a list of names");
    } else if (keyword == "additionalProperties") {
      if (value.is_boolean()) continue;
      if (absl::Status status = CheckKeywords(value, at); !status.ok()) {
        return status;
      }
    } else if (keyword == "enum") {
      if (!value.is_array()) return NotUnderstood(at, "is not a list");
    } else if (keyword == "minimum" || keyword == "maximum") {
      if (!value.is_number()) return NotUnderstood(at, "is not a number");
    } else {
      return NotUnderstood(at, "uses a keyword");
    }
  }
  return absl::OkStatus();
}

// Returns InvalidArgument if `value`, named `name` in messages, is or holds a
// number that is not finite: JSON has no NaN or infinity, though a Struct, read
// from binary, may.
absl::Status CheckFinite(const Json& value, const std::string& name) {
  if (value.is_number_float() && !std::isfinite(value.get<double>())) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " is not a finite number."));
  }
  if (value.is_object()) {
    for (const auto& member : value.items()) {
      if (absl::Status status = CheckFinite(
              member.value(), absl::StrCat(name, ".", member.key()));
          !status.ok()) {
        return status;
      }
    }
  } else if (value.is_array()) {
    for (std::size_t i = 0; i < value.size(); ++i) {
      if (absl::Status status =
              CheckFinite(value[i], absl::StrCat(name, "[", i, "]"));
          !status.ok()) {
        return status;
      }
    }
  }
  return absl::OkStatus();
}

// Whether `value`, a finite number if a number, is of the JSON Schema type
// `type`. An integer is a number with no fraction, as JSON Schema counts it:
// 2.0 is one.
bool HasType(const Json& value, std::string_view type) {
  if (type == "null") return value.is_null();
  if (type == "boolean") return value.is_boolean();
  if (type == "object") return value.is_object();
  if (type == "array") return value.is_array();
  if (type == "number") return value.is_number();
  if (type == "string") return value.is_string();
  if (!value.is_number()) return false;  // "integer"
  if (!value.is_number_float()) return true;
  const double number = value.get<double>();
  return std::trunc(number) == number;
}

// Checks `value`, named `name` in messages, against `schema`, whose keywords
// CheckKeywords has found understood. Every number in `value` is finite, as
// CheckFinite has found.
absl::Status Check(const Json& schema, const Json& value,
                   const std::string& name) {
  if (const auto type = schema.find("type"); type != schema.end()) {
    bool matches = false;
    if (type->is_string()) {
      matches = HasType(value, type->get_ref<const std::string&>());
    } else {
      for (const Json& each : *type) {
        matches = matches || HasType(value, each.get_ref<const std::string&>());
      }
    }
    if (!matches) {
      return absl::InvalidArgumentError(absl::StrCat(name, " is ", Show(value),
                                                     "; it must be of type ",
                                                     Show(*type), "."));
    }
  }
  if (const auto values = schema.find("enum"); values != schema.end()) {
    bool listed = false;
    for (const Json& each : *values) listed = listed || each == value;
    if (!listed) {
      return absl::InvalidArgumentError(absl::StrCat(name, " is ", Show(value),
                                                     "; it must be one of ",
                                                     Show(*values), "."));
    }
  }
  if (value.is_number()) {
    const double number = value.get<double>();
    if (const auto minimum = schema.find("minimum");
        minimum != schema.end() && number < minimum->get<double>()) {
      return absl::InvalidArgumentError(absl::StrCat(name, " is ", Show(value),
                                                     "; it must be at least ",
                                                     Show(*minimum), "."));
    }
    if (const auto maximum = schema.find("maximum");
        maximum != schema.end() && number > maximum->get<double>()) {
      return absl::InvalidArgumentError(absl::StrCat(name, " is ", Show(value),
                                                     "; it must be at most ",
                                                     Show(*maximum), "."));
    }
  }
  if (!value.is_object()) return absl::OkStatus();
  if (const auto required = schema.find("required"); required != schema.end()) {
    for (const Json& member : *required) {
      if (!value.contains(member.get_ref<const std::string&>())) {
        return absl::InvalidArgumentError(
            absl::StrCat(name, " lacks ", member.get_ref<const std::string&>(),
                         ", which is required."));
      }
    }
  }
  const auto properties = schema.find("properties");
  const auto additional = schema.find("additionalProperties");
  for (const auto& member : value.items()) {
    const std::string member_name = absl::StrCat(name, ".", member.key());
    if (properties != schema.end()) {
      if (const auto property = properties->find(member.key());
          property != properties->end()) {
        if (absl::Status status = Check(*property, member.value(), member_name);
            !status.ok()) {
          return status;
        }
        continue;
      }
    }
    if (additional == schema.end()) continue;
    if (additional->is_boolean()) {
      if (additional->get<bool>()) continue;
      return absl::InvalidArgumentError(
          absl::StrCat(name, " has ", member.key(),
                       ", which is not an option of this runtime."));
    }
    if (absl::Status status = Check(*additional, member.value(), member_name);
        !status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateOptions(std::string_view schema, const Struct& options,
                             std::string_view field) {
  absl::StatusOr<Json> parsed = ParseJson(schema);
  if (!parsed.ok()) {
    return absl::InternalError(
        absl::StrCat("The runtime's schema for ", field, " is not JSON."));
  }
  if (absl::Status status = CheckKeywords(*parsed, "#"); !status.ok()) {
    return status;
  }
  const Json value = ToJson(options);
  if (absl::Status status = CheckFinite(value, std::string(field));
      !status.ok()) {
    return status;
  }
  return Check(*parsed, value, std::string(field));
}

}  // namespace model_debugger::runner::service
