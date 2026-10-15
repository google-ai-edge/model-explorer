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

/**
 * Chat-level (conversation) options shared by the editor and the captured configuration overview.
 * They map onto CaptureJob.conversation plus maxOutputTokens and apply when the Conversation is
 * created (src/contracts/SPEC.md); changing one starts a new Chat.
 */
export const GENERATION_GROUPS = [
  {name: 'Prompt', keys: ['systemPrompt', 'applyPromptTemplate']},
  {name: 'Sampling', keys: ['samplerType', 'temperature', 'topK', 'topP', 'seed']},
  {name: 'Thinking', keys: ['thinking', 'thinkingBudget']},
  {name: 'Output', keys: ['maxOutputTokens', 'speculativeDecoding']},
];
export const GENERATION_LABELS: Record<string, string> = {
  samplerType: 'Sampler',
  temperature: 'Temperature',
  topK: 'Top K',
  topP: 'Top P',
  seed: 'Seed',
  thinking: 'Thinking',
  thinkingBudget: 'Thinking budget',
  maxOutputTokens: 'Max output tokens',
  systemPrompt: 'System prompt',
  applyPromptTemplate: 'Prompt template',
  speculativeDecoding: 'Speculative decoding',
};
/** Members saved as 'on' / 'off', or omitted for the model default. */
export const GENERATION_SWITCHES = ['thinking', 'applyPromptTemplate', 'speculativeDecoding'];
/** samplerType values of the contract; a connected Runner narrows them (capability sampler_types). */
export const SAMPLER_TYPES = ['greedy', 'topK', 'topP'];
/** Members a PyTorch run does not take (pytorch_worker._settings). */
export const PYTORCH_UNSUPPORTED = [
  'thinking',
  'thinkingBudget',
  'samplerType',
  'applyPromptTemplate',
  'speculativeDecoding',
];

/** Same bounds as server generation_config.py; no speculative model capabilities. */
export const GENERATION_RANGES: Record<string, {min: number; max: number; step: number | string}> =
  {
    temperature: {min: 0, max: 100, step: 'any'},
    topK: {min: 1, max: 2147483647, step: 1},
    topP: {min: 0, max: 1, step: 'any'},
    seed: {min: 0, max: 4294967295, step: 1},
    thinkingBudget: {min: -1, max: 2147483647, step: 1},
    maxOutputTokens: {min: 1, max: 256, step: 1},
  };
export function switchLabel(value: unknown) {
  return value === 'on' ? 'Enabled' : value === 'off' ? 'Disabled' : 'Model default';
}
