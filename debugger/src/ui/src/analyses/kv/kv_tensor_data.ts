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

import type {KvApi} from '../../server/kv_api';
import type {KvHeadEvidence, KvPositionAlignment} from '../../server/contracts/kv';

export interface KvHeadSelection {
  turn: number;
  contextId: string;
  layer: number;
  kind: string;
  position: number;
  alignment?: KvPositionAlignment | null;
}

export interface KvTensorHead {
  label: string;
  head: number;
  evidence: KvHeadEvidence | null;
  error: string;
}

/**
 * Reads every head of a layer and kind at one position, in one request, for both the channel
 * plot and the numerical comparison table; never substitutes a different source.
 */
export async function readTensorHeads(
  api: Pick<KvApi, 'kvHead'>,
  capture: string,
  selection: KvHeadSelection,
  signal: AbortSignal,
): Promise<KvTensorHead[]> {
  const value = await api.kvHead(
    capture,
    selection.turn,
    selection.contextId,
    selection.layer,
    selection.kind,
    selection.position,
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
    value.heads.some((evidence, head) => evidence.head !== head)
  ) {
    throw new Error('Tensor response does not match the selected source.');
  }
  return value.heads.map((evidence) => ({
    label: `Head ${evidence.head}`,
    head: evidence.head,
    evidence,
    error: '',
  }));
}
