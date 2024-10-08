// Copyright 2024 The AI Edge Model Explorer Authors.
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
// =============================================================================

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_STATUS_MACROS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_STATUS_MACROS_H_

#include "absl/base/optimization.h"
#include "absl/status/status.h"
// Evaluates an expression that produces a `absl::Status`.
// If the status is not ok, returns it from the current function.
//
// For example:
//   absl::Status MultiStepFunction() {
//     RETURN_IF_ERROR(Function(args...));
//     RETURN_IF_ERROR(foo.Method(args...));
//     return absl::OkStatus();
//   }
#define RETURN_IF_ERROR(expr)                                              \
  if (tooling::visualization_client::status_macro_internal:: \
          StatusAdaptorForMacros status_macro_internal_adaptor = {expr}) { \
  } else                                                                   \
    return status_macro_internal_adaptor.Consume()

// Executes an expression `rexpr` that returns a
// `absl::StatusOr<T>`. On OK, extracts its value into the
// variable defined by `lhs`, otherwise returns from the current function. If
// there is an error, `lhs` is not evaluated; thus any side effects that `lhs`
// may have only occur in the success case.
//
// Interface:
//
//   ASSIGN_OR_RETURN(lhs, rexpr)
//
// WARNING: expands into multiple statements; it cannot be used in a single
// statement (e.g. as the body of an if statement without {})!
//
// Example: Declaring and initializing a new variable (ValueType can be anything
//          that can be initialized with assignment, including references):
//   ASSIGN_OR_RETURN(ValueType value, MaybeGetValue(arg));
//
// Example: Assigning to an existing variable:
//   ValueType value;
//   ASSIGN_OR_RETURN(value, MaybeGetValue(arg));
//
// Example: Assigning to an expression with side effects:
//   MyProto data;
//   ASSIGN_OR_RETURN(*data.mutable_str(), MaybeGetValue(arg));
//   // No field "str" is added on error.
//
// Example: Assigning to a std::unique_ptr.
//   ASSIGN_OR_RETURN(std::unique_ptr<T> ptr, MaybeGetPtr(arg));
//
#define ASSIGN_OR_RETURN(lhs, rexpr)    \
  STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_( \
      STATUS_MACROS_IMPL_CONCAT_(_status_or_value, __LINE__), lhs, rexpr)

// =================================================================
// == Implementation details, do not rely on anything below here. ==
// =================================================================

#define STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                                         \
  if (ABSL_PREDICT_FALSE(!statusor.ok())) {                        \
    return statusor.status();                                      \
  }                                                                \
  lhs = std::move(statusor).value()

// Internal helper for concatenating macro values.
#define STATUS_MACROS_IMPL_CONCAT_INNER_(x, y) x##y
#define STATUS_MACROS_IMPL_CONCAT_(x, y) STATUS_MACROS_IMPL_CONCAT_INNER_(x, y)

namespace tooling {
namespace visualization_client {
namespace status_macro_internal {
// Provides a conversion to bool so that it can be used inside an if statement
// that declares a variable.
class StatusAdaptorForMacros {
 public:
  StatusAdaptorForMacros(const absl::Status& status) : status_(status) {}

  StatusAdaptorForMacros(absl::Status&& status) : status_(std::move(status)) {}

  StatusAdaptorForMacros(const StatusAdaptorForMacros&) = delete;
  StatusAdaptorForMacros& operator=(const StatusAdaptorForMacros&) = delete;

  explicit operator bool() const { return status_.ok(); }

  absl::Status&& Consume() { return std::move(status_); }

 private:
  absl::Status status_;
};
}  // namespace status_macro_internal
}  // namespace visualization_client
}  // namespace tooling

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_STATUS_MACROS_H_
