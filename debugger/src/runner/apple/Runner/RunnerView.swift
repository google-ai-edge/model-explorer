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

import SwiftUI
import UniformTypeIdentifiers

#if os(iOS)
  import UIKit
#else
  import AppKit
#endif

struct RunnerView: View {
  @ObservedObject var model: RunnerModel
  @State private var importing = false
  @State private var importIsJob = true
  @State private var exporting = false

  var body: some View {
    RunnerWebView(state: model.uiState, onAction: handle)
      #if os(macOS)
        .frame(minWidth: 640, idealWidth: 980, minHeight: 420, idealHeight: 680)
      #endif
      .fileImporter(isPresented: $importing, allowedContentTypes: importIsJob ? [.json] : [.data]) {
        result in
        switch result {
        case .success(let url): model.importFile(url, isJob: importIsJob)
        case .failure(let error): model.reportError(error)
        }
      }
      #if os(iOS)
        .sheet(isPresented: $exporting) {
          if let url = model.completedRun { FolderExporter(url: url) }
        }
      #endif
  }

  private func handle(_ action: String, _ payload: [String: Any]) {
    // The host enforces capabilities too; disabled HTML buttons are not authority.
    guard model.allowedActions[action] == true else { return }
    switch action {
    case "refreshEnvironment": model.refreshEnvironment()
    case "importJob":
      importIsJob = true
      importing = true
    case "importModel":
      importIsJob = false
      importing = true
    case "run": model.start()
    case "stop": model.stop()
    case "stopGeneration": model.requestStopGeneration()
    case "endSession": model.requestEndSession()
    case "exportCapture":
      #if os(macOS)
        if let url = model.completedRun { NSWorkspace.shared.activateFileViewerSelecting([url]) }
      #else
        exporting = true
      #endif
    #if os(macOS)
      case "refreshInterfaces": model.listenAddresses = RunnerPlatform.addresses()
      case "exportConnection": model.exportConnection()
      case "setAddress":
        guard let address = payload["address"] as? String,
          model.listenAddresses.contains(where: { $0.0 == address })
        else { return }
        model.listenHost = address
        model.changeAddress()
    #endif
    default: break
    }
  }
}

#if os(iOS)
  struct FolderExporter: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
      UIDocumentPickerViewController(forExporting: [url], asCopy: true)
    }
    func updateUIViewController(
      _ uiViewController: UIDocumentPickerViewController, context: Context
    ) {}
  }

#endif
