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

#include "model_debugger/runner/runner_c.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "model_debugger/runner/runner.h"
#include "model_debugger/runner/util/json.h"
#include "model_debugger/v1/runner.pb.h"

// The handle is the Runner itself.
struct ModelDebuggerRunner {
  std::unique_ptr<model_debugger::runner::Runner> runner;
};

namespace {

// Copies `text` into `buffer`, NUL-terminated, cut to `size` - 1 characters.
// Returns text's length.
size_t Copy(std::string_view text, char* buffer, size_t size) {
  if (buffer == nullptr || size == 0) return text.size();
  const size_t copied = text.size() < size - 1 ? text.size() : size - 1;
  std::memcpy(buffer, text.data(), copied);
  buffer[copied] = '\0';
  return text.size();
}

// The platform as a window names it.
std::string PlatformName(model_debugger::v1::HostInfo::Platform platform) {
  switch (platform) {
    case model_debugger::v1::HostInfo::MACOS:
      return "macOS";
    case model_debugger::v1::HostInfo::IOS:
      return "iOS";
    case model_debugger::v1::HostInfo::LINUX:
      return "Linux";
    case model_debugger::v1::HostInfo::ANDROID:
      return "Android";
    default:
      return "";
  }
}

}  // namespace

extern "C" {

ModelDebuggerRunner* model_debugger_runner_start(const char* listen,
                                                 char* error,
                                                 size_t error_size) {
  absl::StatusOr<std::unique_ptr<model_debugger::runner::Runner>> runner =
      model_debugger::runner::Runner::Start(listen != nullptr ? listen : "");
  if (!runner.ok()) {
    Copy(runner.status().message(), error, error_size);
    return nullptr;
  }
  return new ModelDebuggerRunner{*std::move(runner)};
}

size_t model_debugger_runner_get_state(ModelDebuggerRunner* runner,
                                       char* buffer, size_t size) {
  if (runner == nullptr) return Copy("", buffer, size);
  const model_debugger::runner::State state = runner->runner->GetState();
  const model_debugger::runner::Json json{
      {"platform", PlatformName(state.service.host.platform())},
      {"os_version", state.service.host.os_version()},
      {"device", state.service.host.has_chip() ? state.service.host.chip()
                                               : state.service.host.model()},
      {"address", state.address},
      {"port", state.port},
      {"client", state.service.client},
      {"model_sha256", state.service.model_sha256},
      {"turns", state.service.turns},
      {"chat_over", state.service.chat_over},
  };
  // Every string here is the Runner's own or a peer address: UTF-8.
  const absl::StatusOr<std::string> text = model_debugger::runner::Dump(json);
  return Copy(text.ok() ? *text : "{}", buffer, size);
}

void model_debugger_runner_delete(ModelDebuggerRunner* runner) {
  delete runner;
}

}  // extern "C"
