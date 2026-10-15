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

import type {ReportApiService} from '../../../data/report_api_service';
import type {KvHeadEvidence} from '../../../data/contracts/kv';
import type {KvHeadSelection} from './kv_head_inspector';

export interface KvTensorHead {
  label: string;
  head: number;
  evidence: KvHeadEvidence | null;
  error: string;
}

/** Read each head once for both the channel plot and the numerical comparison table.
 * Bound concurrency, preserve head order, and never substitute a different source.
 */
export async function readTensorHeads(
  api: Pick<ReportApiService, 'kvHead'>,
  capture: string,
  selection: KvHeadSelection,
  count: number,
  signal: AbortSignal,
): Promise<KvTensorHead[]> {
  const rows: KvTensorHead[] = Array.from({length: count}, (_, head) => ({
    label: `Head ${head}`,
    head,
    evidence: null,
    error: '',
  }));
  let next = 0;
  await Promise.all(
    Array.from({length: Math.min(4, count)}, async () => {
      while (!signal.aborted && next < count) {
        const row = rows[next++];
        try {
          const value = await api.kvHead(
            capture,
            selection.turn,
            selection.contextId,
            selection.layer,
            selection.kind,
            selection.position,
            row.head,
            null,
            signal,
            selection.alignment,
          );
          const actual = value.selection;
          if (
            actual.turn !== selection.turn ||
            actual.context_id !== selection.contextId ||
            actual.layer !== selection.layer ||
            actual.kind !== selection.kind ||
            actual.position !== selection.position ||
            actual.head !== row.head
          )
            throw new Error('Tensor response does not match the selected source.');
          if (!signal.aborted) row.evidence = value;
        } catch (error) {
          if (!signal.aborted)
            row.error = error instanceof Error ? error.message : 'Unable to load head.';
        }
      }
    }),
  );
  return rows;
}
