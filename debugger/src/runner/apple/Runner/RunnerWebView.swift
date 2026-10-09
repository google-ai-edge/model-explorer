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
import WebKit

/// The same bundled UI runs on every platform. Only this adapter knows WebKit.
@MainActor struct RunnerWebView {
  let state: RunnerUIState
  let onAction: (String, [String: Any]) -> Void

  @MainActor final class Coordinator: NSObject, WKScriptMessageHandler, WKNavigationDelegate {
    var state: RunnerUIState
    var onAction: (String, [String: Any]) -> Void
    weak var webView: WKWebView?
    private var ready = false
    private var lastSnapshot: Data?
    private var publishing = false
    private var generation = UUID()
    private var recoveryAttempts = 0
    private var showingFailure = false
    private let pageURL = Bundle.main.url(
      forResource: "index", withExtension: "html", subdirectory: "ui")

    init(_ parent: RunnerWebView) {
      state = parent.state
      onAction = parent.onAction
    }

    func makeView() -> WKWebView {
      let configuration = WKWebViewConfiguration()
      configuration.websiteDataStore = .default()
      configuration.userContentController.add(self, name: "runner")
      configuration.userContentController.addUserScript(
        WKUserScript(
          source:
            "window.RunnerHost = Object.freeze({postMessage: message => window.webkit.messageHandlers.runner.postMessage(message)});",
          injectionTime: .atDocumentStart, forMainFrameOnly: true))
      let view = WKWebView(frame: .zero, configuration: configuration)
      view.navigationDelegate = self
      #if DEBUG
        view.isInspectable = true
      #endif
      webView = view
      if let pageURL {
        view.loadFileURL(pageURL, allowingReadAccessTo: pageURL.deletingLastPathComponent())
      } else {
        showFailure()
      }
      return view
    }

    func userContentController(
      _ userContentController: WKUserContentController, didReceive message: WKScriptMessage
    ) {
      guard let pageURL, message.frameInfo.isMainFrame,
        message.frameInfo.request.url?.standardizedFileURL == pageURL.standardizedFileURL,
        let body = message.body as? String, body.utf8.count < 4096,
        let data = body.data(using: .utf8),
        let command = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
        command["version"] as? Int == 1, let action = command["action"] as? String
      else { return }
      if action == "ready" {
        ready = true
        publishing = false
        generation = UUID()
        lastSnapshot = nil
        publish()
      } else {
        onAction(action, command)
      }
    }

    func publish() {
      guard ready, !publishing, let webView,
        let data = try? JSONEncoder().encode(state), data != lastSnapshot,
        let object = try? JSONSerialization.jsonObject(with: data)
      else { return }
      publishing = true
      let currentGeneration = generation
      // Arguments are serialized by WebKit; model output is never executable source.
      webView.callAsyncJavaScript(
        "window.RunnerUI.update(state)", arguments: ["state": object],
        in: nil, in: .page
      ) { [weak self] result in
        guard let self, self.generation == currentGeneration else { return }
        self.publishing = false
        switch result {
        case .success:
          self.lastSnapshot = data
          self.recoveryAttempts = 0
          self.publish()  // Coalesce updates received while WebKit was rendering.
        case .failure: self.recoverRenderer()
        }
      }
    }

    private func recoverRenderer() {
      ready = false
      publishing = false
      lastSnapshot = nil
      generation = UUID()
      guard recoveryAttempts == 0, let pageURL else {
        showFailure()
        return
      }
      recoveryAttempts += 1
      webView?.loadFileURL(pageURL, allowingReadAccessTo: pageURL.deletingLastPathComponent())
    }

    private func showFailure() {
      showingFailure = true
      ready = false
      webView?.loadHTMLString(
        "<html><head><meta name='viewport' content='width=device-width,initial-scale=1'></head><body><h1>Runner interface unavailable</h1><p>Close and reopen this window to reload the interface.</p><p>The native runtime is managed independently.</p></body></html>",
        baseURL: nil)
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) { recoverRenderer() }

    func webView(
      _ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
      decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
    ) {
      let url = navigationAction.request.url
      let bundled = pageURL != nil && url?.standardizedFileURL == pageURL?.standardizedFileURL
      decisionHandler(
        bundled || (showingFailure && url?.absoluteString == "about:blank") ? .allow : .cancel)
    }

    func dismantle() {
      ready = false
      generation = UUID()
      webView?.configuration.userContentController.removeScriptMessageHandler(forName: "runner")
      webView?.navigationDelegate = nil
    }
  }

  func makeCoordinator() -> Coordinator { Coordinator(self) }
}

#if os(macOS)
  extension RunnerWebView: NSViewRepresentable {
    func makeNSView(context: Context) -> WKWebView { context.coordinator.makeView() }
    func updateNSView(_ view: WKWebView, context: Context) {
      context.coordinator.state = state
      context.coordinator.onAction = onAction
      context.coordinator.publish()
    }
    static func dismantleNSView(_ view: WKWebView, coordinator: Coordinator) {
      coordinator.dismantle()
    }
  }
#else
  extension RunnerWebView: UIViewRepresentable {
    func makeUIView(context: Context) -> WKWebView { context.coordinator.makeView() }
    func updateUIView(_ view: WKWebView, context: Context) {
      context.coordinator.state = state
      context.coordinator.onAction = onAction
      context.coordinator.publish()
    }
    static func dismantleUIView(_ view: WKWebView, coordinator: Coordinator) {
      coordinator.dismantle()
    }
  }
#endif
