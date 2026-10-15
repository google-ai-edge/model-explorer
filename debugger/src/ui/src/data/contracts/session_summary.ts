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

/** Requested run options; these fields do not prove accelerator residency. */
/** Session-level options as saved: text values, '' for the runtime default (session_rules.py RUN_FIELDS). */
export interface RuntimeOptions {
  backend?: string;
  precision?: string;
  cpuThreads?: string;
  audioBackend?: string;
  audioCpuThreads?: string;
  visionBackend?: string;
  activationDataType?: string;
  prefillChunkSize?: string;
  contextLength?: string;
  enableYnnpack?: string;
  speculativeDecoding?: string;
  parallelFileSectionLoading?: string;
  singleThreadedExecution?: string;
  useRingbuffersLocalAttention?: string;
  gpuDecodeStepsPerSync?: string;
  gpuWaitForWeightUploads?: string;
  gpuMetalResidencySet?: string;
  loraRank?: string;
  audioLoraRank?: string;
}
export type SessionExecutionPhase =
  | 'inactive'
  | 'starting'
  | 'active'
  | 'ending'
  | 'ended'
  | 'interrupted'
  /** The Session still owns an execution slot that failed; close it before starting again. */
  | 'unavailable';
export interface RunnerOwner {
  serverId: string;
  serverName?: string;
  sessionId: string;
  sessionName?: string;
  runId?: string;
  isCurrentServer?: boolean;
}
export interface SessionRunner {
  role: string;
  deviceId: string;
  deviceName?: string;
  buildId?: string;
  phase: string;
  connected: boolean | null;
  lastSeen?: string;
  owner?: RunnerOwner;
}
export interface SessionExecution {
  phase: SessionExecutionPhase;
  runners: SessionRunner[];
  error?: string;
  newChatAllowed?: boolean;
  activeChatId?: string | null;
  activeJobId?: string | null;
  activeOperation?: string | null;
}
export interface SessionSummary {
  id: string;
  parent_session_id?: string;
  generation?: Record<string, Record<string, string | number | null>>;
  name: string;
  created_at: string | null;
  model: string;
  status:
    | 'saved'
    | 'draft'
    | 'preparing'
    | 'initializing'
    | 'ready'
    | 'running'
    | 'finalizing'
    | 'failed'
    | 'cancelled';
  initialized?: boolean;
  execution?: SessionExecution;
  tap_profile?: string;
  tap_points?: Record<string, string[]>;
  tap_prepared?: boolean;
  job_id?: string;
  has_capture: boolean;
  /** The Session ran and ended; it never starts a Runner again (duplicate it to run again). */
  execution_ended?: boolean;
  /** 'imported': created from Runner CLI bundles; its captures are read-only and it never starts a Runner. */
  origin?: 'imported';
  runs: (RuntimeOptions & {
    id: string;
    runtime?: string;
    backend?: string;
    precision?: string;
    artifact?: string;
    source?: string;
    device?: string;
    runnerBuild?: string;
    repository?: string;
    revision?: string;
    sourceUrl?: string;
  })[];
  notice: string;
}
export interface SessionListing {
  sessions: SessionSummary[];
  configuration_template: SessionSummary;
  capabilities: {create: boolean; rename: boolean; duplicate: boolean; delete: boolean};
  unavailable_reason: string;
  generation_available?: boolean;
}

export type SessionConfig = Pick<
  SessionSummary,
  'name' | 'model' | 'runs' | 'tap_profile' | 'tap_points' | 'generation'
>;
