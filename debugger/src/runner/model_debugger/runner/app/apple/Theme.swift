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

import SwiftUI

/// The Model Debugger's colors, as its web pages define them, one set for
/// light and one for dark.
struct Theme {
  let background: Color
  let surface: Color
  let text: Color
  let secondary: Color
  let muted: Color
  let accent: Color
  let accentWash: Color
  let negative: Color
  let negativeWash: Color

  static let light = Theme(
    background: Color(hex: 0xFFFFFF), surface: Color(hex: 0xF0F4F9), text: Color(hex: 0x1F1F1F),
    secondary: Color(hex: 0x444746), muted: Color(hex: 0x62676F),
    accent: Color(hex: 0x0B57D0), accentWash: Color(hex: 0xD3E3FD),
    negative: Color(hex: 0xB3261E), negativeWash: Color(hex: 0xFCE8E6))

  static let dark = Theme(
    background: Color(hex: 0x131314), surface: Color(hex: 0x1E1F20), text: Color(hex: 0xE3E3E3),
    secondary: Color(hex: 0xC4C7C5), muted: Color(hex: 0xA0A5AD),
    accent: Color(hex: 0xA8C7FA), accentWash: Color(hex: 0x0842A0),
    negative: Color(hex: 0xF2B8B5), negativeWash: Color(hex: 0x482523))

  static func current(_ scheme: ColorScheme) -> Theme {
    scheme == .dark ? dark : light
  }
}

extension Color {
  init(hex: UInt32) {
    self.init(
      red: Double((hex >> 16) & 0xFF) / 255, green: Double((hex >> 8) & 0xFF) / 255,
      blue: Double(hex & 0xFF) / 255)
  }
}
