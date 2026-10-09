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
// =============================================================================

import Foundation

@main
struct NativeSmoke {
  static func main() {
    do { try run() } catch {
      fputs(error.localizedDescription + "\n", stderr)
      exit(1)
    }
  }

  static func run() throws {
    let args = CommandLine.arguments
    guard args.count == 4 else {
      throw RunnerError("Usage: native-smoke job.json model.litertlm NEW_OUTPUT_DIRECTORY")
    }
    let job = try JSONDecoder().decode(
      CaptureJob.self, from: Data(contentsOf: URL(fileURLWithPath: args[1])))
    let worker = NativeRunner()
    let result = try worker.execute(
      job: job, model: URL(fileURLWithPath: args[2]),
      destination: URL(fileURLWithPath: args[3]), progress: { print($0) }, delta: { print($0) })
    print(
      "Verified \(result.tensors.count) tensors via Swift -> C API. Platform: \(result.platform)")
  }
}
