/**
 * @license
 * Copyright 2026 The AI Edge Model Explorer Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

/** Stable plot geometry across heatmaps and line charts. */
export function rangeChartGeometry(height: number, dual: boolean) {
  const margin = {l: 60, r: 16, t: dual ? 30 : 12, b: dual ? 172 : 140, autoexpand: false};
  const plotHeight = Math.max(1, height - margin.t - margin.b);
  const fraction = (1 - (dual ? 48 / plotHeight : 0)) / 2;
  return {margin, plotHeight, fraction};
}

/** Reserve enough space for layer rows without changing height with the view. */
export function rangeChartHeight(layers: number, planes = 1) {
  const {margin} = rangeChartGeometry(0, planes === 2);
  return Math.max(
    320,
    Math.max(140, layers * 12) * planes + (planes - 1) * 48 + margin.t + margin.b,
  );
}
