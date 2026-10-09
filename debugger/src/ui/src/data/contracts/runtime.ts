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

import type {RunnerOwner} from './session_summary';

/** Mirrors model_debugger_contracts.jobs.TERMINAL_STATUSES on the server. */
export const TERMINAL_STATUSES = ['completed', 'failed', 'cancelled'] as const;
export type TerminalStatus = (typeof TERMINAL_STATUSES)[number];
export const isTerminal = (status: string): status is TerminalStatus =>
  (TERMINAL_STATUSES as readonly string[]).includes(status);

export interface RuntimeCapability {
  id: string;
  available: boolean;
  reason: string;
  backends: string[];
  supported_options?: string[];
  precisions?: string[];
  contextLengths?: number[];
  /** Largest maxOutputTokens the runtime's Runner accepts (native Runners: 32). */
  max_output_tokens?: number;
  prompt_limit_bytes?: number;
}
export interface RuntimeModel {
  artifact: string;
  name: string;
  runtime?: string;
}
export interface RuntimeCapabilities {
  available: boolean;
  reason: string;
  models: RuntimeModel[];
  backends?: string[];
  runtimes?: RuntimeCapability[];
  supported_options?: string[];
  tap_profiles?: {id: string; name: string; description: string}[];
  platform?: string;
  capture?: string;
  upload_limit_bytes?: number;
  upload_idle_timeout_seconds?: number;
}
export interface RuntimeJob {
  prompt?: string;
  id: string;
  session_id: string;
  operation: 'prepare' | 'initialize' | 'generate' | 'new_chat';
  status: string;
  error: string;
  sequence: number;
  output: Record<string, string>;
  turn: number;
  progress?: string;
}
/** One SSE frame from GET /api/jobs/{id}/events (jobs.py _event/_emit). */
export interface RuntimeJobEvent {
  sequence: number;
  jobId: string;
  sessionId: string;
  turnId?: number;
  type: string;
  status?: string;
  error?: string;
  error_code?: string | null;
  message?: string;
  runId?: string;
  text?: string;
  output?: string;
}
export interface DeviceCandidates {
  devices: RunnerDevice[];
  notice?: string;
}
export interface RunnerDevice {
  id: string;
  name: string;
  runnerId?: string;
  platform?: string;
  transport?: string;
  status?:
    | 'unknown'
    | 'checking'
    | 'ready'
    | 'connected'
    | 'busy'
    | 'unreachable'
    | 'unsupported'
    | 'unpaired';
  checkedAt?: string;
  reason?: string;
  runnerState?: 'not_running' | 'idle' | 'active' | 'unknown';
  runner?: {
    owner?: RunnerOwner;
    buildId?: string;
    build?: RunnerBuild;
    protocolVersion?: number;
    runtimes?: {id: string; backends: string[]; transport: string}[];
    capabilities?: {
      runtime: string;
      backends: string[];
      contextLengths?: number[];
    };
    environment?: {
      chip?: string;
      modelIdentifier?: string;
      architecture?: string;
      physicalMemoryBytes?: number;
    };
  };
}
export interface RunnerBuild {
  id: string;
  label: string;
  version?: string;
  cl?: string;
  platform?: string;
  available?: boolean;
  reason?: string;
}
export interface RunnerBuildCatalog {
  deviceId: string;
  builds: RunnerBuild[];
  canLaunch: boolean;
  reason?: string;
}
