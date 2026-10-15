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

package com.google.ai.edge.modeldebugger.runner

import androidx.compose.ui.graphics.Color

/** The Model Debugger's colors, as its web pages define them. */
data class Theme(
  /** The color behind everything. */
  val background: Color,
  /** The fill of a panel on [background]. */
  val surface: Color,
  /** The most prominent text. */
  val text: Color,
  /** Text that is less prominent than [text]. */
  val secondary: Color,
  /** Labels and separators, fainter than [secondary]. */
  val muted: Color,
  /** The color that marks what is active. */
  val accent: Color,
  /** The accent's pale fill, behind an accent symbol. */
  val accentWash: Color,
  /** The color that marks a failure. */
  val negative: Color,
  /** The negative color's pale fill, behind a failure. */
  val negativeWash: Color,
) {
  companion object {
    /** The colors on a light background. */
    val Light =
      Theme(
        background = Color(0xFFFFFFFF),
        surface = Color(0xFFF0F4F9),
        text = Color(0xFF1F1F1F),
        secondary = Color(0xFF444746),
        muted = Color(0xFF62676F),
        accent = Color(0xFF0B57D0),
        accentWash = Color(0xFFD3E3FD),
        negative = Color(0xFFB3261E),
        negativeWash = Color(0xFFFCE8E6),
      )
    /** The colors on a dark background. */
    val Dark =
      Theme(
        background = Color(0xFF131314),
        surface = Color(0xFF1E1F20),
        text = Color(0xFFE3E3E3),
        secondary = Color(0xFFC4C7C5),
        muted = Color(0xFFA0A5AD),
        accent = Color(0xFFA8C7FA),
        accentWash = Color(0xFF0842A0),
        negative = Color(0xFFF2B8B5),
        negativeWash = Color(0xFF482523),
      )
  }
}
