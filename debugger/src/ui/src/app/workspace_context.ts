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

import type {SessionSummary} from '../data/contracts/session_summary';

/** Navigation and saved capture records have separate identities. */
export interface WorkspaceCaptureContext {
  parentSessionId: string | null;
  activeChatId: string;
  captureRecordId: string | null;
  parentSession: SessionSummary | null;
  activeRecord: SessionSummary | null;
}

export function resolveWorkspaceContext(
  parentSessionId: string | null,
  activeChatId: string,
  records: readonly SessionSummary[],
): WorkspaceCaptureContext {
  const parentSession =
    records.find(
      (record) => record.id === parentSessionId && !record.parent_session_id,
    ) ?? null;
  const activeRecord = !parentSession
    ? null
    : activeChatId === 'capture'
      ? parentSession
      : (records.find(
          (record) =>
            record.id === activeChatId &&
            record.parent_session_id === parentSession.id,
        ) ?? null);
  return {
    parentSessionId,
    activeChatId,
    captureRecordId: activeRecord?.id ?? null,
    parentSession,
    activeRecord,
  };
}
