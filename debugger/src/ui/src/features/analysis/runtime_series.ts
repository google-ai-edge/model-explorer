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

export type AnalysisRuntime = 'ref' | 'target';

const RUNTIME_SERIES = {
  ref: {label: 'Ref', name: 'Reference', color: 'var(--me-primary-color)', dash: 'solid'},
  target: {
    label: 'Target',
    name: 'Target',
    color: 'var(--me-on-surface-variant-color)',
    dash: 'dash',
  },
} as const;

export function runtimeSeries(runtime: AnalysisRuntime) {
  return RUNTIME_SERIES[runtime];
}

/** Shared fill for selected-token cells and their legend swatches. */
export function runtimeSelectionBackground(runtime: AnalysisRuntime) {
  return runtime === 'ref'
    ? 'light-dark(#d2eada, #426e51)'
    : 'light-dark(#fadbd6, #8b4c49)';
}
