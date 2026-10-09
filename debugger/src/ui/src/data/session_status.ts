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
  SessionExecutionPhase,
  SessionSummary,
} from './contracts/session_summary';

/** The execution lifetime and captured-data status are separate server fields. */
export function sessionExecutionPhase(
  session: SessionSummary,
): SessionExecutionPhase {
  return (
    session.execution?.phase ??
    (session.initialized
      ? 'active'
      : session.status === 'initializing'
        ? 'starting'
        : 'inactive')
  );
}

export function sessionStatusLabel(session: SessionSummary): string {
  const phase = sessionExecutionPhase(session);
  return phase === 'inactive'
    ? session.status.charAt(0).toUpperCase() + session.status.slice(1)
    : (
        {
          starting: 'Starting',
          active: 'Active',
          ending: 'Ending',
          ended: 'Ended',
          interrupted: 'Interrupted',
          unavailable: 'Unavailable',
        } as const
      )[phase];
}

/** The one execution question the UI answers per Session: is its Model Server on? */
export type ModelServerState = 'failure' | 'starting' | 'on' | 'off';
export const MODEL_SERVER_LABELS: Record<ModelServerState, string> = {
  failure: 'Failure',
  starting: 'Starting',
  on: 'On',
  off: 'Off',
};
/** Preparing the capture model is part of starting. An ending Session still holds its devices,
 * so it stays on until they are released. A request the Server refused reads as a failure. */
export function modelServerState(
  session: SessionSummary,
  requestError = '',
): ModelServerState {
  const phase = sessionExecutionPhase(session);
  if (phase === 'active' || phase === 'ending') return 'on';
  if (
    phase === 'starting' ||
    session.status === 'preparing' ||
    session.status === 'initializing'
  )
    return 'starting';
  return phase === 'interrupted' ||
    phase === 'unavailable' ||
    session.status === 'failed' ||
    requestError
    ? 'failure'
    : 'off';
}
/** A Session can be entered when there is something to do in it: a Model Server that is on,
 * or captured data to read. */
export function sessionEnterable(session: SessionSummary): boolean {
  return modelServerState(session) === 'on' || session.has_capture === true;
}

export function sessionIsActive(session: SessionSummary): boolean {
  // 'unavailable' still owns an execution slot; it must be closed before starting again.
  return ['starting', 'active', 'ending', 'unavailable'].includes(
    sessionExecutionPhase(session),
  );
}

export function sessionConfigurationLocked(session: SessionSummary): boolean {
  return (
    sessionIsActive(session) ||
    ['preparing', 'initializing', 'running', 'finalizing'].includes(
      session.status,
    )
  );
}

export function copiedSessionName(name: string): string {
  return name.trim().slice(0, 73).trimEnd() + ' (copy)';
}
