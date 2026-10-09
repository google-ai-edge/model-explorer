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

/** Shared generation groups used by the editor and captured configuration overview. */
export const GENERATION_GROUPS = [
  {name: 'System prompt', keys: ['systemPrompt']},
  {name: 'Sampling', keys: ['temperature', 'topK', 'topP', 'seed']},
  {name: 'Thinking', keys: ['thinking', 'thinkingBudget']},
  {name: 'Output', keys: ['maxOutputTokens']},
];
export const GENERATION_LABELS: Record<string, string> = {
  temperature: 'Temperature',
  topK: 'Top K',
  topP: 'Top P',
  seed: 'Seed',
  thinking: 'Thinking',
  thinkingBudget: 'Thinking budget',
  maxOutputTokens: 'Max output tokens',
  systemPrompt: 'System prompt',
};

/** Same bounds as server generation_config.py; no speculative model capabilities. */
export const GENERATION_RANGES: Record<
  string,
  {min: number; max: number; step: number | string}
> = {
  temperature: {min: 0, max: 100, step: 'any'},
  topK: {min: 1, max: 2147483647, step: 1},
  topP: {min: 0, max: 1, step: 'any'},
  seed: {min: 0, max: 4294967295, step: 1},
  thinkingBudget: {min: -1, max: 2147483647, step: 1},
  maxOutputTokens: {min: 1, max: 256, step: 1},
};
