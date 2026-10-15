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

// macOS and iOS: every fact from sysctl.

#include "model_debugger/runner/service/host_info.h"

#include <TargetConditionals.h>
#include <sys/sysctl.h>
#include <sys/types.h>

#include <cstdint>
#include <optional>
#include <string>

#include "model_debugger/v1/runner.pb.h"

namespace model_debugger::runner::service {
namespace {

// Returns the text of sysctl `name`, or nullopt if the system has none.
std::optional<std::string> SysctlText(const char* name) {
  size_t size = 0;
  if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) {
    return std::nullopt;
  }
  std::string text(size, '\0');
  if (sysctlbyname(name, text.data(), &size, nullptr, 0) != 0) {
    return std::nullopt;
  }
  // The size counts the terminating NUL.
  while (!text.empty() && text.back() == '\0') text.pop_back();
  return text;
}

// Returns the integer of sysctl `name`, or nullopt if the system has none.
template <typename T>
std::optional<T> SysctlNumber(const char* name) {
  T value{};
  size_t size = sizeof(value);
  if (sysctlbyname(name, &value, &size, nullptr, 0) != 0 ||
      size != sizeof(value)) {
    return std::nullopt;
  }
  return value;
}

}  // namespace

v1::HostInfo MeasureHost(bool can_run_on_gpu) {
  v1::HostInfo host;
#if TARGET_OS_IPHONE
  host.set_platform(v1::HostInfo::IOS);
#else
  host.set_platform(v1::HostInfo::MACOS);
#endif
  if (const auto version = SysctlText("kern.osproductversion")) {
    host.set_os_version(*version);
  }
  if (const auto build = SysctlText("kern.osversion")) {
    host.set_os_build(*build);
  }
#if defined(__aarch64__)
  host.set_architecture("arm64");
#else
  host.set_architecture("x86_64");
#endif
#if TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR
  // On an iOS device hw.model is the board, such as "D83AP"; the model
  // identifier, such as "iPhone16,1", is hw.machine. A simulator runs on the
  // Mac, and reports the Mac's.
  const char* const model_name = "hw.machine";
#else
  const char* const model_name = "hw.model";
#endif
  if (const auto model = SysctlText(model_name)) {
    host.set_model(*model);
  }
  const auto chip = SysctlText("machdep.cpu.brand_string");
  if (chip) host.set_chip(*chip);
  if (can_run_on_gpu) {
    // On Apple silicon the GPU is the chip's; the system reports no other
    // name for it.
    host.mutable_gpu()->set_name(chip.value_or(""));
  }
  if (const auto cpus = SysctlNumber<int32_t>("hw.ncpu")) {
    host.set_cpu_count(*cpus);
  }
  if (const auto memory = SysctlNumber<int64_t>("hw.memsize")) {
    host.set_memory_bytes(*memory);
  }
  return host;
}

}  // namespace model_debugger::runner::service
