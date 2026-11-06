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

import type {
  AnalysisColumns,
  CapturedToken,
  Session,
  StageAnalysisId,
  StagePositions,
} from '../../server/contracts/capture';
import type {ColumnSteps} from '../../widgets/format/step_labels';

/** Which of a token's step texts a column shows: the step that produced it (Final Logits, the
 *  intermediate tensors) or the one that wrote its K and V (the KV cache). */
export type StepFact = 'produced' | 'written';

/** A side's token of a turn at its own context position, input or output. */
function tokenAt(
  session: Session | null,
  turn: number,
  side: 'ref' | 'target',
  position: number | null,
): CapturedToken | undefined {
  if (position === null) {
    return undefined;
  }
  const conversation = session?.conversation?.find((c) => c.turn === turn && c.run === side);
  return [...(conversation?.input_tokens ?? []), ...(conversation?.tokens ?? [])].find(
    (token) => token.position === position,
  );
}

function text(token: CapturedToken | undefined, fact: StepFact): string | null {
  return (fact === 'produced' ? token?.produced_text : token?.written_text) ?? null;
}

/** Why a column's two sides read different tokens yet compare, in the server's words: from the
 *  analysis's `context` column on, at a column with a token on both sides; null elsewhere. */
export function columnContext(
  columns: AnalysisColumns | null | undefined,
  column: number,
  twoSided: boolean,
): string | null {
  const context = columns?.context;
  return twoSided && context && column >= context.start ? context.reason.text : null;
}

/** Which step a column of a turn's stage is on each side, in the server's words: the token each
 *  side reads at the column (the alignment's pair, else the column's position on both sides);
 *  and, after the divergence, why the analysis's two sides there still compare. */
export function columnSteps(
  session: Session | null,
  turn: number | null,
  column: number,
  alignment: StagePositions | null,
  fact: StepFact,
  analysis?: StageAnalysisId,
): ColumnSteps | null {
  if (turn === null) {
    return null;
  }
  const [ref, target] = alignment
    ? (alignment.pairs[column - alignment.start] ?? [null, null])
    : [column, column];
  const context = columnContext(
    analysis ? alignment?.columns?.[analysis] : null,
    column,
    ref !== null && target !== null,
  );
  return {
    ref: text(tokenAt(session, turn, 'ref', ref), fact),
    target: text(tokenAt(session, turn, 'target', target), fact),
    ...(context ? {context} : {}),
  };
}

/** Which step a Final Logits row is on each side: its generated tokens, by their steps; and
 *  `context`, the server's words why its logits read different tokens yet compare. */
export function generatedSteps(
  session: Session | null,
  turn: number | null,
  ref: number | null | undefined,
  target: number | null | undefined,
  context?: string | null,
): ColumnSteps | null {
  if (turn === null) {
    return null;
  }
  const token = (
    side: 'ref' | 'target',
    step: number | null | undefined,
  ): CapturedToken | undefined =>
    step == null
      ? undefined
      : session?.conversation?.find((c) => c.turn === turn && c.run === side)?.tokens?.[step];
  return {
    ref: text(token('ref', ref), 'produced'),
    target: text(token('target', target), 'produced'),
    ...(context ? {context} : {}),
  };
}
