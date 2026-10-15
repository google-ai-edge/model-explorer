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

export type ChannelReduction = 'min' | 'max' | 'mean';
export interface ChannelBin extends KvRangeBin {
  ref: number | null;
  target: number | null;
}
/** Choose a real element of a displayed bin. Without a trace side, prefer Ref, then Target.
 * Match bin validity, use the first index for Mean, and break extrema ties by index. */
export function channelRepresentative(
  rows: readonly KvHeadChannel[],
  bin: KvRangeBin,
  reduction: ChannelReduction,
  side?: 'ref' | 'target',
): number | null {
  const values = rows.filter((row) => row.channel >= bin.start && row.channel < bin.end);
  if (values.length !== bin.end - bin.start || !values.length) return null;
  for (const runtime of side ? [side] : (['ref', 'target'] as const)) {
    if (values.some((row) => typeof row[runtime] !== 'number' || !Number.isFinite(row[runtime])))
      continue;
    if (reduction === 'mean') return bin.start;
    const representative = values.reduce((best, row) => {
      const value = row[runtime] as number,
        previous = best[runtime] as number;
      return (reduction === 'min' ? value < previous : value > previous) ||
        (value === previous && row.channel < best.channel)
        ? row
        : best;
    });
    return representative.channel;
  }
  return null;
}
export function channelBinSize(range: KvRangeBin, capacity: number) {
  return Math.max(1, Math.ceil((range.end - range.start) / Math.max(1, capacity)));
}
/** Head dimension bins summarize each runtime separately, never a comparison metric.
 * Incomplete bins stay gaps: a missing/nonfinite channel must not become a valid mean. */
export function channelBins(
  rows: readonly KvHeadChannel[],
  range: KvRangeBin,
  capacity: number,
  reduction: ChannelReduction,
): ChannelBin[] {
  const size = channelBinSize(range, capacity);
  const bins: {start: number; end: number; rows: KvHeadChannel[]}[] = [];
  for (let start = range.start; start < range.end; start += size)
    bins.push({start, end: Math.min(range.end, start + size), rows: []});
  for (const row of rows) {
    if (row.channel >= range.start && row.channel < range.end)
      bins[Math.floor((row.channel - range.start) / size)].rows.push(row);
  }
  return bins.map(({start, end, rows}) => {
    const reduce = (side: 'ref' | 'target') => {
      if (
        rows.length !== end - start ||
        rows.some((r) => typeof r[side] !== 'number' || !Number.isFinite(r[side]))
      )
        return null;
      const values = rows.map((r) => r[side] as number);
      if (reduction === 'mean') return values.reduce((sum, v) => sum + v / values.length, 0);
      return values.reduce((a, b) => (reduction === 'min' ? Math.min(a, b) : Math.max(a, b)));
    };
    return {start, end, ref: reduce('ref'), target: reduce('target')};
  });
}
