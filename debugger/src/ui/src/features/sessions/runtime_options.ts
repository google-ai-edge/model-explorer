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
 * Session-level (engine) options of a run. They are requests, not captured values, and map
 * onto CaptureJob.engine plus the fixed backend and contextLength (src/contracts/SPEC.md).
 * Every value is saved as text: digits, an enum value, or '' / 'on' / 'off' for a switch.
 */
import type {RuntimeOptions} from '../../data/contracts/session_summary';
export type RuntimeOptionKind = 'backend' | 'threads' | 'context' | 'enum' | 'int' | 'switch';
export interface RuntimeOptionField {
  key: RuntimeOptionKey;
  label: string;
  kind: RuntimeOptionKind;
  /** Only meaningful with this text backend; disabled and cleared otherwise. */
  backend?: 'CPU' | 'GPU';
  options?: readonly string[];
  help: string;
}
export const runtimeOptionKeys = [
  'backend',
  'cpuThreads',
  'audioBackend',
  'audioCpuThreads',
  'visionBackend',
  'activationDataType',
  'prefillChunkSize',
  'contextLength',
  'enableYnnpack',
  'speculativeDecoding',
  'parallelFileSectionLoading',
  'singleThreadedExecution',
  'useRingbuffersLocalAttention',
  'gpuDecodeStepsPerSync',
  'gpuWaitForWeightUploads',
  'gpuMetalResidencySet',
  'loraRank',
  'audioLoraRank',
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
export const activationDataTypes = ['float32', 'float16', 'int16', 'int8'] as const;
/** Members beyond the backend, thread and context settings; shown as "Advanced engine settings". */
export const advancedRuntimeOptionFields: readonly RuntimeOptionField[] = [
  {
    key: 'enableYnnpack',
    label: 'YNNPACK',
    kind: 'switch',
    help: 'Delegate supported operations to YNNPACK before XNNPACK (set_enable_ynnpack).',
  },
  {
    key: 'speculativeDecoding',
    label: 'Speculative decoding (engine)',
    kind: 'switch',
    help: 'Engine-level speculative decoding. The Runner keeps it off unless enabled; the debugger pairs one released token per step.',
  },
  {
    key: 'parallelFileSectionLoading',
    label: 'Parallel section loading',
    kind: 'switch',
    help: 'Load the sections of the .litertlm file in parallel (runtime default: on).',
  },
  {
    key: 'singleThreadedExecution',
    label: 'Single-threaded execution',
    kind: 'switch',
    help: 'Run the engine on a single thread (set_single_threaded_execution).',
  },
  {
    key: 'useRingbuffersLocalAttention',
    label: 'Ring-buffer local attention',
    kind: 'switch',
    help: 'Use ring buffers for local attention; ignored with a warning on unsupported models or backends.',
  },
  {
    key: 'gpuDecodeStepsPerSync',
    label: 'GPU decode steps per sync',
    kind: 'int',
    backend: 'GPU',
    help: 'Decode steps between GPU synchronizations (1–1024). GPU backend only.',
  },
  {
    key: 'gpuWaitForWeightUploads',
    label: 'GPU wait for weight uploads',
    kind: 'switch',
    backend: 'GPU',
    help: 'Wait for weight uploads before running. GPU backend only.',
  },
  {
    key: 'gpuMetalResidencySet',
    label: 'GPU Metal residency set',
    kind: 'switch',
    backend: 'GPU',
    help: 'Keep GPU resources resident through a Metal residency set. GPU backend only.',
  },
  {
    key: 'loraRank',
    label: 'LoRA rank',
    kind: 'int',
    help: 'Rank of the text LoRA adapter the model was prepared with (0–1024).',
  },
  {
    key: 'audioLoraRank',
    label: 'Audio LoRA rank',
    kind: 'int',
    help: 'Rank of the audio LoRA adapter (0–1024).',
  },
];
export const runtimeOptionFields: readonly {key: RuntimeOptionKey; label: string}[] = [
  {key: 'backend', label: 'Text Backend'},
  {key: 'cpuThreads', label: 'Text CPU threads'},
  {key: 'audioBackend', label: 'Audio Backend'},
  {key: 'audioCpuThreads', label: 'Audio CPU threads'},
  {key: 'visionBackend', label: 'Vision Backend'},
  {key: 'activationDataType', label: 'Activation dtype'},
  {key: 'prefillChunkSize', label: 'Prefill chunk size'},
  {key: 'contextLength', label: 'Context length'},
  ...advancedRuntimeOptionFields.map(({key, label}) => ({key, label})),
];
export const switchRuntimeOptionKeys = advancedRuntimeOptionFields
  .filter((field) => field.kind === 'switch')
  .map((field) => field.key);
export function runtimeOptionValue(run: RuntimeOptions, key: RuntimeOptionKey): string {
  if (
    (key === 'cpuThreads' && run.backend !== 'CPU') ||
    (key === 'audioCpuThreads' && run.audioBackend !== 'CPU')
  )
    return '—';
  const value = run[key];
  if ((switchRuntimeOptionKeys as readonly string[]).includes(key))
    return value === 'on' ? 'On' : value === 'off' ? 'Off' : 'Default';
  return String(value || 'Default');
}
