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

import CryptoKit
import Foundation
import SwiftUI

#if os(iOS)
  import UIKit
#else
  import AppKit
  import Network
  import Darwin
#endif

/// The only platform-specific storage, power and connection configuration.
@MainActor
enum RunnerPlatform {
  static var runnerSlot: String {
    let args = ProcessInfo.processInfo.arguments
    guard let index = args.firstIndex(of: "--runner-slot"), args.indices.contains(index + 1),
      args[index + 1] == "target"
    else { return "ref" }
    return "target"
  }
  static var listenPort: UInt16 { runnerSlot == "target" ? 8770 : 8769 }
  #if os(macOS)
    static var deviceStorage: URL {
      FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Model Debugger Runner", isDirectory: true)
    }
  #endif
  static var storage: URL {
    #if os(iOS)
      return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    #else
      return runnerSlot == "ref"
        ? deviceStorage : deviceStorage.appendingPathComponent("slots/target", isDirectory: true)
    #endif
  }

  #if os(macOS)
    private static var activity: NSObjectProtocol?
  #endif
  static func keepAwake(_ active: Bool) {
    #if os(iOS)
      UIApplication.shared.isIdleTimerDisabled = active
    #else
      if active && activity == nil {
        activity = ProcessInfo.processInfo.beginActivity(
          options: [.idleSystemSleepDisabled, .userInitiated], reason: "Model Debugger Session")
      } else if !active, let previous = activity {
        ProcessInfo.processInfo.endActivity(previous)
        activity = nil
      }
    #endif
  }

  #if os(macOS)
    static var connectionURL: URL { storage.appendingPathComponent("connection.json") }

    static func connectionConfig() throws -> [String: Any] {
      if runnerSlot == "target" {
        let base = try deviceConnectionConfig()
        try FileManager.default.createDirectory(at: storage, withIntermediateDirectories: true)
        let previous = (try? Data(contentsOf: connectionURL)).flatMap {
          try? JSONSerialization.jsonObject(with: $0) as? [String: Any]
        }
        let token = SHA256.hash(data: Data(((base["token"] as? String ?? "") + ":target").utf8)).map
        { String(format: "%02x", $0) }.joined()
        let value: [String: Any] = [
          "id": previous?["id"] as? String ?? UUID().uuidString.lowercased(),
          "deviceId": base["id"] ?? "", "name": base["name"] ?? "Mac Runner", "slot": "target",
          "host": base["host"] ?? "127.0.0.1", "port": 8770, "token": token,
        ]
        try saveConnection(value)
        return value
      }
      return try deviceConnectionConfig()
    }

    private static func deviceConnectionConfig() throws -> [String: Any] {
      let connectionURL = deviceStorage.appendingPathComponent("connection.json")
      try FileManager.default.createDirectory(at: deviceStorage, withIntermediateDirectories: true)
      // Both build processes share the same physical identity, including on
      // first launch. A cross-process lock protects credential creation.
      let descriptor = Darwin.open(
        deviceStorage.appendingPathComponent("configuration.lock").path, O_CREAT | O_RDWR, 0o600)
      guard descriptor >= 0 else { throw RunnerError("Could not lock device configuration.") }
      defer {
        flock(descriptor, LOCK_UN)
        Darwin.close(descriptor)
      }
      guard flock(descriptor, LOCK_EX) == 0 else {
        throw RunnerError("Could not lock device configuration.")
      }
      try FileManager.default.createDirectory(at: storage, withIntermediateDirectories: true)
      if FileManager.default.fileExists(atPath: connectionURL.path) {
        guard
          let value = try JSONSerialization.jsonObject(with: Data(contentsOf: connectionURL))
            as? [String: Any]
        else {
          throw RunnerError("Invalid connection configuration.")
        }
        return value
      }
      var bytes = [UInt8](repeating: 0, count: 32)
      guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else {
        throw RunnerError("Could not create connection credentials.")
      }
      let config: [String: Any] = [
        "id": UUID().uuidString.lowercased(), "name": Host.current().localizedName ?? "Mac Runner",
        "host": "127.0.0.1", "port": 8769,
        "token": bytes.map { String(format: "%02x", $0) }.joined(),
      ]
      try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
        .write(to: connectionURL, options: .atomic)
      try FileManager.default.setAttributes(
        [.posixPermissions: 0o600], ofItemAtPath: connectionURL.path)
      return config
    }

    static func saveConnection(_ config: [String: Any]) throws {
      try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
        .write(to: connectionURL, options: .atomic)
      try FileManager.default.setAttributes(
        [.posixPermissions: 0o600], ofItemAtPath: connectionURL.path)
    }

    /// Offer actual Thunderbolt bridge IPv4 addresses, plus loopback. Do not
    /// silently expose the Runner on every interface or fall back to Wi-Fi.
    static func addresses() -> [(String, String)] {
      var values = [("127.0.0.1", "This Mac only")]
      var head: UnsafeMutablePointer<ifaddrs>?
      guard getifaddrs(&head) == 0 else { return values }
      defer { freeifaddrs(head) }
      var current = head
      while let item = current {
        defer { current = item.pointee.ifa_next }
        let name = String(cString: item.pointee.ifa_name)
        guard name.hasPrefix("bridge"), let address = item.pointee.ifa_addr,
          address.pointee.sa_family == UInt8(AF_INET)
        else { continue }
        var buffer = [CChar](repeating: 0, count: Int(NI_MAXHOST))
        if getnameinfo(
          address, socklen_t(address.pointee.sa_len), &buffer, socklen_t(buffer.count), nil, 0,
          NI_NUMERICHOST) == 0
        {
          let ip = String(cString: buffer)
          values.append((ip, "\(name) · \(ip)"))
        }
      }
      return values
    }

    static func exportConnection() throws {
      let panel = NSSavePanel()
      panel.nameFieldStringValue = "model-debugger-connection.json"
      panel.message =
        "Import this connection file on the Mac running the Model Debugger server. It contains the access key."
      if panel.runModal() == .OK, let url = panel.url {
        // Registration describes the physical Mac; both execution slots
        // derive their endpoints from these shared device credentials.
        try Data(contentsOf: deviceStorage.appendingPathComponent("connection.json")).write(
          to: url, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
      }
    }
  #endif

  /// Every compatible build uses this physical-device lease before activation.
  /// Session resets retain the lease; only completed model cleanup releases it.
  static func claimDevice(serverID: String, sessionID: String, role: String) throws {
    #if os(macOS)
      guard role == runnerSlot else {
        throw RunnerError("Runner slot does not match the requested side.")
      }
      try updateDeviceLease { owners in
        guard
          owners.values.allSatisfy({
            $0["serverId"] as? String == serverID && $0["sessionId"] as? String == sessionID
          })
        else {
          throw RunnerError("Device is occupied by another Session.")
        }
        if let existing = owners[runnerSlot], existing["pid"] as? Int != Int(getpid()) {
          throw RunnerError("Runner slot is already occupied.")
        }
        owners[runnerSlot] = ["pid": Int(getpid()), "serverId": serverID, "sessionId": sessionID]
      }
    #endif
  }

  static func releaseDevice() {
    #if os(macOS)
      try? updateDeviceLease { owners in
        if owners[runnerSlot]?["pid"] as? Int == Int(getpid()) {
          owners.removeValue(forKey: runnerSlot)
        }
      }
    #endif
  }

  #if os(macOS)
    private static func updateDeviceLease(_ change: (inout [String: [String: Any]]) throws -> Void)
      throws
    {
      try FileManager.default.createDirectory(at: deviceStorage, withIntermediateDirectories: true)
      let descriptor = Darwin.open(
        deviceStorage.appendingPathComponent("device-lease.lock").path, O_CREAT | O_RDWR, 0o600)
      guard descriptor >= 0 else { throw RunnerError("Could not lock physical device.") }
      defer {
        flock(descriptor, LOCK_UN)
        Darwin.close(descriptor)
      }
      guard flock(descriptor, LOCK_EX) == 0 else {
        throw RunnerError("Could not lock physical device.")
      }
      let url = deviceStorage.appendingPathComponent("device-lease.json")
      var owners: [String: [String: Any]] = [:]
      if FileManager.default.fileExists(atPath: url.path) {
        guard
          let value = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
            as? [String: [String: Any]]
        else {
          throw RunnerError("Physical device ownership is unknown.")
        }
        owners = value.filter { _, value in
          guard let pid = value["pid"] as? Int, pid > 0 else { return true }
          return kill(pid_t(pid), 0) == 0 || errno != ESRCH
        }
      }
      try change(&owners)
      try JSONSerialization.data(withJSONObject: owners, options: [.sortedKeys]).write(
        to: url, options: .atomic)
      try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }
  #endif
}
