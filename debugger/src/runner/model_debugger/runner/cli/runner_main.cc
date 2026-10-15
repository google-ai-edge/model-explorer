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

// The Runner as a command: serves model_debugger.v1.Runner until the process
// is stopped.

#include <iostream>
#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/status/statusor.h"
#include "model_debugger/runner/runner.h"

ABSL_FLAG(std::string, listen, "127.0.0.1:8769",
          "HOST:PORT to serve on; port 0 picks a free one.");

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage("Serves a Model Debugger runner.");
  absl::ParseCommandLine(argc, argv);
  absl::StatusOr<std::unique_ptr<model_debugger::runner::Runner>> runner =
      model_debugger::runner::Runner::Start(absl::GetFlag(FLAGS_listen));
  if (!runner.ok()) {
    std::cerr << "runner: " << runner.status().message() << "\n";
    return 1;
  }
  std::cout << "runner: listening on port " << (*runner)->GetState().port
            << "\n"
            << std::flush;
  (*runner)->Wait();
  return 0;
}
