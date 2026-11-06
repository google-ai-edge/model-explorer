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

// A runtime's options, checked against the JSON Schema the runtime reports for
// them in RuntimeInfo: one check for every runtime, made before the runtime
// sees the options.

#ifndef MODEL_DEBUGGER_RUNNER_SERVICE_OPTIONS_SCHEMA_H_
#define MODEL_DEBUGGER_RUNNER_SERVICE_OPTIONS_SCHEMA_H_

#include <string_view>

#include "google/protobuf/struct.pb.h"
#include "absl/status/status.h"

namespace model_debugger::runner::service {

// Checks `options`, the request field named `field` (such as
// "LoadRequest.runtime_options"), against `schema`, a JSON Schema (draft
// 2020-12) as a runtime reports it.
//
// Understands the keywords the runtimes' schemas use, and no other: type,
// properties, required, additionalProperties, enum, minimum and maximum. The
// whole schema is checked for them on every call, whatever `options` holds:
// a schema that uses another keyword fails every call, rather than leaving a
// rule unchecked.
//
// Returns InvalidArgument naming the option and the rule it breaks; a number
// that is not finite, which JSON cannot hold, breaks every schema, wherever it
// is in `options`, even where the schema says nothing. Returns
// Internal if `schema` is not JSON, or uses a keyword, or a form of one, that
// is not understood.
absl::Status ValidateOptions(std::string_view schema,
                             const google::protobuf::Struct& options,
                             std::string_view field);

}  // namespace model_debugger::runner::service

#endif  // MODEL_DEBUGGER_RUNNER_SERVICE_OPTIONS_SCHEMA_H_
