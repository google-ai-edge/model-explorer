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

/// Facts the host application derives from its bundle. Core never reads `Bundle.main`;
/// the Runner App installs these at launch, and CLIs and tests keep the empty defaults.
struct HostBundleFacts {
  var isAppBundle = false
  var privateFrameworksURL: URL? = nil
  var runtimeBuildURL: URL? = nil
  var appVersion: String? = nil
  var appBuild: String? = nil
  var buildID: String? = nil
  var buildCL: String? = nil
  var buildLabel: String? = nil

  init() {}

  init(bundle: Bundle) {
    isAppBundle = bundle.bundleURL.pathExtension == "app"
    privateFrameworksURL = bundle.privateFrameworksURL
    runtimeBuildURL = bundle.url(forResource: "runtime-build", withExtension: "json")
    appVersion = bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
    appBuild = bundle.object(forInfoDictionaryKey: "CFBundleVersion") as? String
    buildID = bundle.object(forInfoDictionaryKey: "ModelDebuggerBuildID") as? String
    buildCL = bundle.object(forInfoDictionaryKey: "ModelDebuggerCL") as? String
    buildLabel = bundle.object(forInfoDictionaryKey: "ModelDebuggerBuildLabel") as? String
  }
}

enum HostBundle {
  private static let lock = NSLock()
  private static var installed = HostBundleFacts()
  static var facts: HostBundleFacts { lock.withLock { installed } }
  /// Install once at launch, before the first environment or backend lookup.
  static func install(_ facts: HostBundleFacts) { lock.withLock { installed = facts } }
}
