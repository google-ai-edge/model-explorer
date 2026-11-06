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

/** Which step a column is on each side, in the server's words; a side without a token there has
 *  none. `context` is why the column's two sides read different tokens yet compare, in the
 *  server's words; absent where they read the same. */
export interface ColumnSteps {
  ref?: string | null;
  target?: string | null;
  context?: string | null;
}

/** The rows that say which step a column is: one when both sides say the same, else one per side
 *  that says anything; then the column's context, where its two sides read different tokens. */
export function stepRows(steps: ColumnSteps | null | undefined): {label: string; value: string}[] {
  const ref = steps?.ref ?? null;
  const target = steps?.target ?? null;
  const context = steps?.context ? [{label: 'Context', value: steps.context}] : [];
  if (ref !== null && ref === target) {
    return [{label: 'Step', value: ref}, ...context];
  }
  return [
    ...(ref !== null ? [{label: 'Reference step', value: ref}] : []),
    ...(target !== null ? [{label: 'Target step', value: target}] : []),
    ...context,
  ];
}
