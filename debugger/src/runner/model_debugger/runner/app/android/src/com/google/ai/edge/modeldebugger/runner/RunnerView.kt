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

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Bolt
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material.icons.outlined.ErrorOutline
import androidx.compose.material.icons.outlined.HourglassEmpty
import androidx.compose.material.icons.outlined.Power
import androidx.compose.material.icons.outlined.StopCircle
import androidx.compose.material.icons.outlined.ViewInAr
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * The Runner's window, as the Model Debugger draws it: under the app bar, the device, a card that
 * says where the Runner stands and what to do, then its facts. Facts not reported yet read as a
 * dash. The page the Apple App draws, in Compose.
 */
@Composable
fun RunnerView(state: RunnerState?, failure: String?) {
  val theme = if (isSystemInDarkTheme()) Theme.Dark else Theme.Light
  val (symbol, title, hint) = standing(state, failure)
  Column(
    modifier =
      Modifier.fillMaxSize().background(theme.background).statusBarsPadding().padding(24.dp),
    verticalArrangement = Arrangement.spacedBy(24.dp),
  ) {
    Row(verticalAlignment = Alignment.CenterVertically) {
      Image(painterResource(R.drawable.logo), contentDescription = null, Modifier.size(30.dp))
      Spacer(Modifier.width(12.dp))
      Text(
        buildAnnotatedString {
          append("Model Debugger ")
          withStyle(SpanStyle(color = theme.muted)) { append("· ") }
          withStyle(SpanStyle(color = theme.accent)) { append("Runner") }
        },
        color = theme.text,
        fontSize = 20.sp,
        fontWeight = FontWeight.SemiBold,
      )
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
      Image(
        painterResource(R.drawable.device_phone),
        contentDescription = "Phone",
        colorFilter = ColorFilter.tint(theme.secondary),
      )
      Spacer(Modifier.width(12.dp))
      Column {
        Text(
          state?.device ?: "—",
          color = theme.text,
          fontSize = 16.sp,
          fontWeight = FontWeight.SemiBold,
        )
        Text(
          state?.let { "${it.platform} ${it.osVersion}" } ?: "—",
          color = theme.muted,
          fontSize = 12.sp,
        )
      }
    }
    Column(
      modifier =
        Modifier.fillMaxWidth().background(theme.surface, RoundedCornerShape(24.dp)).padding(32.dp),
      horizontalAlignment = Alignment.CenterHorizontally,
    ) {
      Box(
        Modifier.size(76.dp)
          .background(
            if (failure == null) theme.accentWash else theme.negativeWash,
            RoundedCornerShape(24.dp),
          ),
        contentAlignment = Alignment.Center,
      ) {
        Icon(
          symbol,
          contentDescription = null,
          Modifier.size(36.dp),
          tint = if (failure == null) theme.accent else theme.negative,
        )
      }
      Spacer(Modifier.size(22.dp))
      Text(
        title,
        color = theme.text,
        fontSize = 24.sp,
        fontWeight = FontWeight.SemiBold,
        textAlign = TextAlign.Center,
      )
      Spacer(Modifier.size(12.dp))
      // The line is there in every state, empty if there is nothing to do,
      // so that the card, and the facts under it, keep their place.
      Text(
        hint ?: " ",
        color = theme.secondary,
        fontSize = 14.sp,
        textAlign = TextAlign.Center,
        modifier = Modifier.widthIn(max = 330.dp),
      )
    }
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
      Fact("Listening", state?.let { "${it.address}:${it.port}" }, theme)
      Fact("Server", state?.client?.ifEmpty { null }, theme)
      Fact("Model", state?.modelSha256?.ifEmpty { null }?.take(12), theme)
      Fact("Turns", state?.takeIf { it.modelSha256.isNotEmpty() }?.turns?.toString(), theme)
    }
  }
}

/** A named fact on one row; a fact not reported yet reads as a dash. */
@Composable
private fun Fact(name: String, value: String?, theme: Theme) {
  Row {
    Text(
      name,
      color = theme.muted,
      fontSize = 12.sp,
      fontFamily = FontFamily.Monospace,
      modifier = Modifier.width(88.dp),
    )
    Text(value ?: "—", color = theme.text, fontSize = 12.sp, fontFamily = FontFamily.Monospace)
  }
}

private data class Standing(val symbol: ImageVector, val title: String, val hint: String? = null)

/** The card's symbol, its title, and what the user should do, if anything. */
private fun standing(state: RunnerState?, failure: String?): Standing =
  when {
    failure != null -> Standing(Icons.Outlined.ErrorOutline, "Failed to start", failure)
    state == null -> Standing(Icons.Outlined.HourglassEmpty, "Starting")
    state.chatOver ->
      Standing(Icons.Outlined.StopCircle, "Chat over", "Start a new session in Model Debugger.")
    state.client.isEmpty() ->
      Standing(
        Icons.Outlined.Power,
        "Waiting for connection",
        "Select this device in Model Debugger.",
      )
    state.modelSha256.isEmpty() -> Standing(Icons.Outlined.CheckCircle, "Connected")
    state.turns == 0 -> Standing(Icons.Outlined.ViewInAr, "Model loaded")
    else -> Standing(Icons.Outlined.Bolt, "Serving")
  }
