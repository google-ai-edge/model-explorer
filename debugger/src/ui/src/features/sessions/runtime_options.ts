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

/** Configuration fields from debugger-shell.html runtimeConfigKeys; these are requests, not captured values. */
import type {RuntimeOptions} from '../../data/contracts/session_summary';
export const runtimeOptionKeys = [
  'backend',
  'cpuThreads',
  'audioBackend',
  'audioCpuThreads',
  'visionBackend',
  'forceF32',
  'prefillBatchSizes',
  'contextLength',
] as const;
export const pytorchRuntimeOptionKeys = [
  'backend',
  'precision',
  'cpuThreads',
  'contextLength',
] as const;
export type RuntimeOptionKey =
  | (typeof runtimeOptionKeys)[number]
  | (typeof pytorchRuntimeOptionKeys)[number];
export const runtimeOptionFields = [
  {key: 'backend', label: 'Text Backend'},
  {key: 'cpuThreads', label: 'Text CPU threads'},
  {key: 'audioBackend', label: 'Audio Backend'},
  {key: 'audioCpuThreads', label: 'Audio CPU threads'},
  {key: 'visionBackend', label: 'Vision Backend'},
  {key: 'forceF32', label: 'Activation dtype'},
  {key: 'prefillBatchSizes', label: 'Prefill batch sizes'},
  {key: 'contextLength', label: 'Context length'},
] as const;
export function runtimeOptionValue(
  run: RuntimeOptions,
  key: RuntimeOptionKey,
): string {
  if (
    (key === 'cpuThreads' && run.backend !== 'CPU') ||
    (key === 'audioCpuThreads' && run.audioBackend !== 'CPU')
  )
    return '—';
  return key === 'forceF32'
    ? run.forceF32
      ? 'FP32'
      : 'Default'
    : String(run[key] || 'Default');
}
