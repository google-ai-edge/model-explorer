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

import {inject, Injectable} from '@angular/core';
import {ServerClient, type UploadOptions} from './server_client';
import type {Session} from './contracts/capture';
import type {RuntimeJob} from './contracts/jobs';
import type {ChatCapabilities} from './contracts/runtime';
import type {
  ChatGeneration,
  SessionConfig,
  SessionListing,
  SessionSummary,
} from './contracts/sessions';

/** /api/sessions: the Session list, its management, the Session lifecycle, its chats, and the
 *  evidence of one chat's capture (/api/session). */
@Injectable({providedIn: 'root'})
export class SessionsApi {
  private readonly client = inject(ServerClient);
  sessions(): Promise<SessionListing> {
    return this.client.get<SessionListing>('sessions');
  }
  manageSession(
    operation: 'create' | 'delete',
    payload: Partial<SessionConfig> & {id?: string},
  ): Promise<SessionSummary> {
    return this.client.post<SessionSummary>(`sessions/${operation}`, payload);
  }
  /** What each side of a Session's chats may set, as the server describes it for the side's
   *  device, runner build and run. */
  chatOptions(id: string): Promise<ChatCapabilities> {
    return this.client.get<ChatCapabilities>(`sessions/chat-options?id=${encodeURIComponent(id)}`);
  }
  renameSession(id: string, name: string): Promise<SessionSummary> {
    return this.client.post<SessionSummary>('sessions/rename', {id, name});
  }
  /** Turns a Session's Model Server off, or cancels its start; answers the Session as listed. */
  close(id: string): Promise<SessionSummary> {
    return this.client.post<SessionSummary>(`sessions/${encodeURIComponent(id)}/close`, {});
  }
  /** Sends a turn to chat `chat`, which the server refuses unless it is the live one (another
   *  page may have started a chat meanwhile). A turn sends only its prompt: how long a reply may
   *  be is the chat's setting. Answers the turn's job. */
  turn(id: string, chat: number, prompt: string): Promise<RuntimeJob> {
    return this.client.post<RuntimeJob>(`sessions/${encodeURIComponent(id)}/turns`, {
      prompt,
      chat,
    });
  }
  /** Starts a new chat of a Session, each side with its settings; answers the job that starts it.
   *  The live chat, if any, ends as it is admitted. */
  startChat(id: string, generation: ChatGeneration): Promise<RuntimeJob> {
    return this.client.post<RuntimeJob>(`sessions/${encodeURIComponent(id)}/chats`, {generation});
  }
  /** Where the browser saves a Session that has ended as one file, `<name>.mdsession`. */
  exportUrl(id: string): string {
    return this.client.url(`sessions/${encodeURIComponent(id)}/export`);
  }
  /** Imports a Session another computer exported; answers it as the list shows it. */
  importSession(file: File, options: UploadOptions = {}): Promise<SessionSummary> {
    return this.client.upload<SessionSummary>('sessions/import', file, options);
  }
  /** The evidence of one chat's capture (capture_id.ts). */
  session(captureId: string | null, signal?: AbortSignal): Promise<Session> {
    return this.client.getCapture<Session>(captureId, 'session', signal);
  }
}
