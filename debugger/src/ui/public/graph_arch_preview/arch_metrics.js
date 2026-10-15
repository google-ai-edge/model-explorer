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

// The host supplies metrics at the selected token position, keyed by semantic output.
// Never spread a layer output value over its internal tensors or block summaries.
window.archMetricData = function(context, scope, anchor, members) {
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const values = {...context?.tensorMetrics?.[scope]};
  const layer = /^decoder_layer_(\d+)$/.exec(scope);
  if (layer && !Object.hasOwn(values, 'output')) values.output = context?.layers?.[Number(layer[1])];
  // A block summary excludes the separately displayed final output by identity.
  const ids = [...new Set(members ?? [anchor])].filter(id => !members || id !== anchor);
  const samples = context?.simulatedMetrics ?? [];
  const simulated = new Map();
  // Reproducible preview values, never override an explicit binding (including null).
  for (const id of new Set([...ids, anchor])) {
    if (Object.hasOwn(values, id) || !samples.length) continue;
    let hash = 2166136261;
    for (const c of `${scope}|${id}|${context?.stage}|${context?.position ?? 0}`) {
      hash = Math.imul(hash ^ c.charCodeAt(0), 16777619) >>> 0;
    }
    const sample = samples[hash % samples.length];
    values[id] = sample.value;
    simulated.set(id, sample);
  }
  const known = ids.map(id => values[id]).filter(finite);
  const mode = members ? (context?.summary === 'min' ? 'min' : 'max') : 'output';
  const value = mode === 'min' ? (known.length ? Math.min(...known) : null)
    : mode === 'max' ? (known.length ? Math.max(...known) : null)
    : finite(values[anchor]) ? values[anchor] : null;
  const sample = mode === 'output' ? simulated.get(anchor) : ids.map(id => simulated.get(id)).find(item => item?.value === value);
  const color = sample ?? (value === null ? null : context?.colors?.[String(value)]);
  const text = value === null ? '—' : sample?.text ?? context?.formatted?.[String(value)] ?? String(value);
  return {
    value, mode, simulated: !!sample, available: known.length, total: ids.length,
    text,
    bg: color?.bgColor ?? 'var(--metric-empty-bg)',
    fg: color?.textColor ?? (value === null ? 'var(--muted)' : 'var(--text)'),
    title: `${members ? (mode === 'min' ? 'Internal Min' : 'Internal Max') : 'Tensor output'} · ${value === null ? 'No metric data' : text}${sample ? ' · Preview sample' : ''}${members && mode !== 'output' ? ` · ${known.length}/${ids.length} outputs available` : ''}`,
  };
};

// Arch tags always show the tensor icon and metric. Names belong to Details.
window.archTensorDisplay = function(metricWidth) {
  return {labelWidth: 18, tagOffset: 27, totalWidth: 27 + metricWidth};
};
