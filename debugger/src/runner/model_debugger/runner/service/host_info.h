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

#ifndef MODEL_DEBUGGER_RUNNER_SERVICE_HOST_INFO_H_
#define MODEL_DEBUGGER_RUNNER_SERVICE_HOST_INFO_H_

#include "model_debugger/v1/runner.pb.h"

namespace model_debugger::runner::service {

// Returns the machine this process runs on, as the system reports it:
// nothing is guessed, and a value the platform does not report is left
// unset. The GPU is set when `can_run_on_gpu`, what the runtime says of
// itself, named as the system names it, by its chip. Measured once, at
// startup; GetInfo reports it.
v1::HostInfo MeasureHost(bool can_run_on_gpu);

}  // namespace model_debugger::runner::service

#endif  // MODEL_DEBUGGER_RUNNER_SERVICE_HOST_INFO_H_
