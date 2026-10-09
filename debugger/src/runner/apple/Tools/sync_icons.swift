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

// Deterministic native icon packaging. The approved transparent PNG and SVG
// are identical to the web logo; this only supplies platform icon backgrounds.
import AppKit
import Foundation

let root = URL(fileURLWithPath: CommandLine.arguments[1])
let catalog = root.appendingPathComponent("Runner/Assets.xcassets")
let source = root.appendingPathComponent("Brand/debugger-logo-1024.png")
let logo = NSImage(contentsOf: source)!
func json(_ value: [String: Any], _ url: URL) throws {
  try FileManager.default.createDirectory(
    at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
  try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys]).write(
    to: url)
}
let info: [String: Any] = ["author": "xcode", "version": 1]
try json(["info": info], catalog.appendingPathComponent("Contents.json"))
let icons = catalog.appendingPathComponent("AppIcon.appiconset")
var entries: [[String: String]] = []
func render(_ size: Int, _ name: String, mac: Bool) throws {
  let bitmap = NSBitmapImageRep(
    bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size, bitsPerSample: 8,
    samplesPerPixel: mac ? 4 : 3, hasAlpha: mac, isPlanar: false, colorSpaceName: .deviceRGB,
    bytesPerRow: 0, bitsPerPixel: 0)!
  NSGraphicsContext.saveGraphicsState()
  NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
  let edge = CGFloat(size)
  let background =
    mac
    ? NSRect(x: edge * 0.08, y: edge * 0.08, width: edge * 0.84, height: edge * 0.84)
    : NSRect(x: 0, y: 0, width: edge, height: edge)
  NSColor.white.setFill()
  NSBezierPath(
    roundedRect: background, xRadius: mac ? edge * 0.18 : 0, yRadius: mac ? edge * 0.18 : 0
  ).fill()
  let inset = edge * (mac ? 0.19 : 0.13)
  logo.draw(in: NSRect(x: inset, y: inset, width: edge - inset * 2, height: edge - inset * 2))
  NSGraphicsContext.restoreGraphicsState()
  try bitmap.representation(using: .png, properties: [:])!.write(
    to: icons.appendingPathComponent(name))
}
try FileManager.default.createDirectory(at: icons, withIntermediateDirectories: true)
try render(1024, "ios-1024.png", mac: false)
entries.append([
  "idiom": "universal", "platform": "ios", "size": "1024x1024", "filename": "ios-1024.png",
])
for size in [16, 32, 128, 256, 512] {
  for scale in [1, 2] {
    let name = "mac-\(size)-\(scale)x.png"
    try render(size * scale, name, mac: true)
    entries.append([
      "idiom": "mac", "size": "\(size)x\(size)", "scale": "\(scale)x", "filename": name,
    ])
  }
}
try json(["info": info, "images": entries], icons.appendingPathComponent("Contents.json"))
