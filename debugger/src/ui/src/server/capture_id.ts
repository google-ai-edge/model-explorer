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
 * The identity of one capture: a chat of a Session. A Session holds several chats, each with its
 * own capture, so a capture is named by both, as one string: every cache and kept view keyed by
 * it is a chat's own, and every capture-scoped request names the two (`session_id=`, `chat=`).
 */

/** The capture of chat `chat` of Session `session`. */
export function captureId(session: string, chat: number): string {
  return `${session}#${chat}`;
}

/** The Session and chat a capture names; null for a string that names none. */
export function parseCaptureId(id: string): {session: string; chat: number} | null {
  const at = id.lastIndexOf('#');
  const chat = Number(id.slice(at + 1));
  return at > 0 && Number.isInteger(chat) && chat > 0 ? {session: id.slice(0, at), chat} : null;
}
