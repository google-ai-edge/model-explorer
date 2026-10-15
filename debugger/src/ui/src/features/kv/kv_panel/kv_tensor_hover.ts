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

import type {KvHeadChannel, KvRangeBin} from '../../../data/contracts/kv';
import {formatMetric} from '../../../shared/format/format';
import {hoverBin, hoverInterval, type ChartHoverContent} from '../../analysis/chart_hover';
import {channelBins, type ChannelReduction} from './kv_channel_bins';

/** Line and Scatter inspect the same raw elements. Bins are reduced separately per runtime. */
export function tensorHover(
  groups: readonly (readonly KvHeadChannel[])[],
  bin: KvRangeBin,
  reduction: ChannelReduction,
  options: {
    dimensionLabel?: string;
    groupLabels?: readonly string[];
    comparisonAllowed?: boolean;
  } = {},
): ChartHoverContent {
  const single = bin.end - bin.start === 1;
  return {
    properties: [
      {label: 'Dimension', value: options.dimensionLabel ?? 'Head dimension'},
      {label: 'Index', value: hoverInterval(bin.start, bin.end)},
      {label: 'Bin', value: hoverBin(bin.start, bin.end, reduction)},
    ],
    columns: single ? ['Ref', 'Target', 'Δ', '|Δ|'] : ['Ref', 'Target'],
    rows: groups.map((rows, index) => {
      const value = channelBins(rows, bin, 1, reduction)[0];
      const delta =
        options.comparisonAllowed !== false && value?.ref != null && value?.target != null
          ? value.target - value.ref
          : null;
      return {
        label:
          options.groupLabels?.[index] ??
          (groups.length === 1 ? 'Value' : index ? 'Value cache' : 'Key cache'),
        values: [
          value?.ref,
          value?.target,
          ...(single ? [delta, delta == null ? null : Math.abs(delta)] : []),
        ].map((v) => formatMetric(v ?? null)),
      };
    }),
  };
}
