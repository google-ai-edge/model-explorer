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
import Network

/// Shared authenticated WebSocket control. iOS uses usbmuxd + loopback;
/// macOS binds only the explicitly selected local interface.
@MainActor
final class AppleRunnerTransport: RunnerTransport {
  typealias State = RunnerConnectionState
  private(set) var connectionID: UUID?
  private var listener: NWListener?
  private var connection: NWConnection?
  private var peers: [ObjectIdentifier: NWConnection] = [:]
  private var handshakeTasks: [ObjectIdentifier: Task<Void, Never>] = [:]
  private var sharesServerFilesystem = false
  nonisolated static let inspectionProtocol = "model-debugger-inspect-v1"
  var isConnected: Bool { connectionID != nil }
  var onMessage: (([String: Any]) -> Void)?
  var onDisconnect: (() -> Void)?
  var onStatus: ((State, String) -> Void)?
  var inspection: (() -> RunnerDescriptor?)?

  func start(documents: URL) throws {
    guard listener == nil else {
      return
    }
    #if os(macOS)
      let config = try RunnerPlatform.connectionConfig()
      let host = config["host"] as? String ?? "127.0.0.1"
      guard RunnerPlatform.addresses().contains(where: { $0.0 == host }) else {
        throw RunnerError("Selected bridge address is unavailable. Choose an active interface.")
      }
    #else
      let host = "127.0.0.1"
      let configURL = documents.appendingPathComponent("usb-bridge.json")
      guard FileManager.default.fileExists(atPath: configURL.path) else {
        onStatus?(.unconfigured, "Connect this device by USB and pair it from Model Debugger.")
        return
      }
      let config =
        try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any] ?? [:]
    #endif
    #if os(macOS)
      sharesServerFilesystem = host == "127.0.0.1"
    #endif
    guard
      let token = config["token"] as? String,
      token.count == 64,
      token.allSatisfy({ $0.isHexDigit })
    else {
      throw RunnerError("Invalid USB connection configuration.")
    }
    let ws = NWProtocolWebSocket.Options()
    ws.autoReplyPing = true
    ws.maximumMessageSize = 1024 * 1024
    ws.setClientRequestHandler(.main) { protocols, headers in
      let authorized = headers.contains {
        $0.name.lowercased() == "authorization" && $0.value == "Bearer " + token
      }
      let browserOrigin = headers.contains { $0.name.lowercased() == "origin" }
      let inspecting = protocols == [Self.inspectionProtocol]
      return NWProtocolWebSocket.Response(
        status: authorized && !browserOrigin && (protocols.isEmpty || inspecting)
          ? .accept : .reject,
        subprotocol: inspecting ? Self.inspectionProtocol : nil)
    }
    guard let port = NWEndpoint.Port(rawValue: RunnerPlatform.listenPort) else {
      throw RunnerError("Invalid Runner listen port.")
    }
    let parameters = NWParameters.tcp
    parameters.defaultProtocolStack.applicationProtocols.insert(ws, at: 0)
    parameters.requiredLocalEndpoint = .hostPort(host: NWEndpoint.Host(host), port: port)
    let server = try NWListener(using: parameters)
    server.stateUpdateHandler = { [weak self, weak server] state in
      Task { @MainActor in
        guard let self, let server, self.listener === server else {
          return
        }
        if case .ready = state {
          self.onStatus?(.waiting, "Waiting for Model Debugger")
        }
        if case .failed(let error) = state {
          self.listener = nil
          self.onStatus?(.failed, error.localizedDescription)
        }
      }
    }
    server.newConnectionHandler = { [weak self, weak server] peer in
      Task { @MainActor in
        guard let self, let server, self.listener === server else {
          peer.cancel()
          return
        }
        self.accept(peer)
      }
    }
    listener = server
    server.start(queue: .main)
  }

  private static var platform: String {
    #if os(macOS)
      return "macOS"
    #else
      return "iOS"
    #endif
  }

  func restart(documents: URL) throws {
    disconnect()
    listener?.cancel()
    listener = nil
    try start(documents: documents)
  }

  private func accept(_ peer: NWConnection) {
    // Inspectors never acquire the control slot. Bound pending/observer peers.
    guard peers.count < 5 else {
      peer.cancel()
      return
    }
    let peerID = ObjectIdentifier(peer)
    peers[peerID] = peer
    peer.stateUpdateHandler = { [weak self, weak peer] state in
      Task { @MainActor in
        guard let self, let peer, self.peers[ObjectIdentifier(peer)] === peer else {
          return
        }
        switch state {
        case .ready:
          self.handshakeTasks.removeValue(forKey: ObjectIdentifier(peer))?.cancel()
          let metadata =
            peer.metadata(definition: NWProtocolWebSocket.definition)
            as? NWProtocolWebSocket.Metadata
          if metadata?.selectedSubprotocol == Self.inspectionProtocol {
            guard
              let descriptor = self.inspection?(),
              let data = try? JSONEncoder().encode(descriptor)
            else {
              self.remove(peer)
              return
            }
            self.send(data, to: peer)
            self.receiveInspection(peer)
            return
          }
          guard self.connection == nil else {
            self.remove(peer)
            return
          }
          self.connection = peer
          self.connectionID = UUID()
          self.onStatus?(.connected, "Model Debugger connected")
          var hello: [String: Any] = [
            "type": "hello",
            "protocolVersion": RunnerCommandCodec.protocolVersion,
            "debuggerEnabled": NativeRunner.debuggerEnabled,
            "platform": Self.platform,
            "fileTransfer": true,
            "fileLink": self.sharesServerFilesystem,
            "runtimes": self.inspection?()?.runtimes.map(\.id) ?? ["LiteRT-LM"],
          ]
          if let descriptor = self.inspection?() {
            hello["runnerInstance"] = descriptor.runnerInstance.uuidString.lowercased()
            hello["buildId"] = descriptor.buildID
            hello["capabilities"] = try? JSONSerialization.jsonObject(
              with: JSONEncoder().encode(descriptor.capabilities))
          }
          self.send(hello)
          self.receive(peer)
        case .failed, .cancelled:
          self.remove(peer)
        default:
          break
        }
      }
    }
    peer.start(queue: .main)
    handshakeTasks[peerID] = Task { @MainActor [weak self, weak peer] in
      // Duration-based Task.sleep(for:) requires macOS 13.0+ / iOS 16.0+, matching
      // the Apple Runner deployment target; evicts peers that do not complete handshake within 5s.
      try? await Task.sleep(for: .seconds(5))
      guard !Task.isCancelled, let self, let peer, self.connection !== peer else {
        return
      }
      self.remove(peer)
    }
  }

  private func receiveInspection(_ peer: NWConnection) {
    // Even an authenticated observer cannot submit execution/file commands.
    peer.receiveMessage { [weak self, weak peer] _, _, _, _ in
      Task { @MainActor in
        if let self, let peer {
          self.remove(peer)
        }
      }
    }
  }

  private func receive(_ peer: NWConnection) {
    peer.receiveMessage { [weak self, weak peer] data, context, _, error in
      Task { @MainActor in
        guard let self, let peer, self.connection === peer else {
          return
        }
        let metadata =
          context?.protocolMetadata(definition: NWProtocolWebSocket.definition)
          as? NWProtocolWebSocket.Metadata
        if error != nil || metadata?.opcode == .close || (data == nil && context == nil) {
          self.disconnect()
          return
        }
        if let data, metadata?.opcode == .text || metadata?.opcode == .binary {
          do {
            if metadata?.opcode == .binary {
              var (message, payload) = try RunnerCommandCodec.decodeFrame(data)
              message["data"] = payload
              self.onMessage?(message)
            } else {
              guard let message = try JSONSerialization.jsonObject(with: data) as? [String: Any]
              else {
                throw RunnerError("Expected a JSON command.")
              }
              self.onMessage?(message)
            }
          } catch {
            self.send(["type": "error", "error": error.localizedDescription])
          }
        }
        self.receive(peer)
      }
    }
  }

  func send(_ value: [String: Any]) {
    guard
      let peer = connection,
      let data = try? JSONSerialization.data(withJSONObject: value)
    else {
      return
    }
    send(data, to: peer)
  }

  func send(_ header: [String: Any], payload: Data) {
    guard
      let peer = connection,
      let frame = try? RunnerCommandCodec.encodeFrame(header, payload: payload)
    else {
      return
    }
    send(frame, to: peer, opcode: .binary)
  }

  private func send(_ data: Data, to peer: NWConnection, opcode: NWProtocolWebSocket.Opcode = .text)
  {
    let context = NWConnection.ContentContext(
      identifier: "frame", metadata: [NWProtocolWebSocket.Metadata(opcode: opcode)])
    peer.send(
      content: data, contentContext: context, isComplete: true,
      completion: .contentProcessed { [weak self, weak peer] error in
        guard error != nil else {
          return
        }
        Task { @MainActor in
          if let self, let peer {
            self.remove(peer)
          }
        }
      })
  }

  func disconnect() {
    for peer in Array(peers.values) {
      remove(peer)
    }
  }

  func shutdown() {
    listener?.cancel()
    listener = nil
    disconnect()
  }

  func sendFinal(_ value: [String: Any], completion: @escaping () -> Void) {
    guard
      let peer = connection,
      let data = try? JSONSerialization.data(withJSONObject: value)
    else {
      completion()
      return
    }
    let context = NWConnection.ContentContext(
      identifier: "final-json", metadata: [NWProtocolWebSocket.Metadata(opcode: .text)])
    peer.send(
      content: data, contentContext: context, isComplete: true,
      completion: .contentProcessed { _ in
        Task { @MainActor in
          completion()
        }
      })
  }

  private func remove(_ peer: NWConnection) {
    let peerID = ObjectIdentifier(peer)
    handshakeTasks.removeValue(forKey: peerID)?.cancel()
    guard peers.removeValue(forKey: peerID) != nil else {
      return
    }
    peer.cancel()
    guard connection === peer else {
      return
    }
    let authenticated = connectionID != nil
    connection = nil
    connectionID = nil
    if authenticated {
      onDisconnect?()
    }
    onStatus?(.waiting, "Model Debugger disconnected")
  }
}
