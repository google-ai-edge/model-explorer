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
 * Browser-local drafts, one per Session, are separate from captured/runtime conversation history.
 * Which Session is open is the page's address, not stored here.
 */
export interface StoredChats {
  drafts: Record<string, string>;
}
/** Returns an empty browser-local draft store. */
export function emptyChats(): StoredChats {
  return {drafts: {}};
}
/** Applied on write and on read; longer drafts are truncated rather than lost. */
export const MAX_DRAFT_LENGTH = 16000;
/**
 * Keep only drafts of Sessions the server still lists, each under its Session's id: a key of an
 * older page's Chat (`<session>:<chat>`) is dropped, and so are empty drafts.
 */
export function pruneStoredChats(stored: StoredChats, sessionIds: Iterable<string>): StoredChats {
  const known = new Set(sessionIds);
  return {
    drafts: Object.fromEntries(
      Object.entries(stored.drafts).filter(([key, value]) => known.has(key) && value !== ''),
    ),
  };
}
/** Parses stored draft JSON from localStorage, returning `emptyChats()` on invalid input. */
export function parseStoredChats(raw: string | null): StoredChats {
  const result = emptyChats();
  try {
    const data = JSON.parse(raw ?? 'null');
    if (data?.version !== 1) {
      return result;
    }
    for (const [key, value] of Object.entries(data.drafts ?? {})) {
      if (typeof value === 'string') {
        result.drafts[key] = value.slice(0, MAX_DRAFT_LENGTH);
      }
    }
  } catch {
    /* Corrupt or unavailable storage must not prevent opening a capture. */
  }
  return result;
}
