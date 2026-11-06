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

import type {KvHeadChannel, KvRangeBin} from '../../server/contracts/kv';
import {formatMetric} from '../format/format';
import {hoverBin, hoverInterval, type ChartHoverContent} from './chart_hover';
import {channelBins, type ChannelBin, type ChannelReduction} from './channel_bins';

export interface TensorHoverOptions {
  dimensionLabel?: string;
  groupLabels?: readonly string[];
  comparisonAllowed?: boolean;
}

/** Line and Scatter inspect the same raw elements. Bins are reduced separately per runtime. */
export function tensorHover(
  groups: readonly (readonly KvHeadChannel[])[],
  bin: KvRangeBin,
  reduction: ChannelReduction,
  options: TensorHoverOptions = {},
): ChartHoverContent {
  return binHover(
    groups.map((rows) => channelBins(rows, bin, 1, reduction)[0]),
    bin,
    reduction,
    options,
  );
}

/** The hover of a bin whose values are already reduced, one per group (`tensorHover`). */
export function binHover(
  values: readonly (ChannelBin | undefined)[],
  bin: KvRangeBin,
  reduction: ChannelReduction,
  options: TensorHoverOptions = {},
): ChartHoverContent {
  const single = bin.end - bin.start === 1;
  return {
    properties: [
      {label: 'Dimension', value: options.dimensionLabel ?? 'Head dimension'},
      {label: 'Index', value: hoverInterval(bin.start, bin.end)},
      {label: 'Bin', value: hoverBin(bin.start, bin.end, reduction)},
    ],
    columns: single ? ['Ref', 'Target', 'Δ', '|Δ|'] : ['Ref', 'Target'],
    rows: values.map((value, index) => {
      const delta =
        options.comparisonAllowed !== false && value?.ref != null && value?.target != null
          ? value.target - value.ref
          : null;
      return {
        label:
          options.groupLabels?.[index] ??
          (values.length === 1 ? 'Value' : index ? 'Value cache' : 'Key cache'),
        values: [
          value?.ref,
          value?.target,
          ...(single ? [delta, delta == null ? null : Math.abs(delta)] : []),
        ].map((v) => formatMetric(v ?? null)),
      };
    }),
  };
}
