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

// Android: the build's facts from system properties, the machine's from the
// kernel.

#include <sys/system_properties.h>
#include <unistd.h>

#include <cstdint>
#include <string>

#include "model_debugger/runner/service/host_info.h"
#include "model_debugger/v1/runner.pb.h"

namespace model_debugger::runner::service {
namespace {

// Returns system property `name`, or empty if the system has none.
std::string Property(const char* name) {
  char value[PROP_VALUE_MAX] = "";
  __system_property_get(name, value);
  return value;
}

}  // namespace

v1::HostInfo MeasureHost(bool can_run_on_gpu) {
  v1::HostInfo host;
  host.set_platform(v1::HostInfo::ANDROID);
  if (std::string version = Property("ro.build.version.release");
      !version.empty()) {
    host.set_os_version(version);
  }
  if (std::string build = Property("ro.build.id"); !build.empty()) {
    host.set_os_build(build);
  }
#if defined(__aarch64__)
  host.set_architecture("arm64");
#else
  host.set_architecture("x86_64");
#endif
  const std::string manufacturer = Property("ro.product.manufacturer");
  const std::string model = Property("ro.product.model");
  if (!model.empty()) {
    host.set_model(manufacturer.empty() ? model : manufacturer + " " + model);
  }
  // The SoC's own name exists from API 31; before, only the board's.
  std::string chip;
  if (std::string soc = Property("ro.soc.model"); !soc.empty()) {
    const std::string vendor = Property("ro.soc.manufacturer");
    chip = vendor.empty() ? soc : vendor + " " + soc;
    host.set_chip(chip);
  }
  if (can_run_on_gpu) {
    // The GPU is the SoC's; the system names it by the SoC.
    host.mutable_gpu()->set_name(chip);
  }
  if (const long cpus = sysconf(_SC_NPROCESSORS_CONF); cpus > 0) {
    host.set_cpu_count(static_cast<int32_t>(cpus));
  }
  if (const long pages = sysconf(_SC_PHYS_PAGES),
      page_size = sysconf(_SC_PAGE_SIZE);
      pages > 0 && page_size > 0) {
    host.set_memory_bytes(static_cast<int64_t>(pages) * page_size);
  }
  return host;
}

}  // namespace model_debugger::runner::service
