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

import {MAX_DRAFT_LENGTH, parseStoredChats, pruneStoredChats} from './chat_storage';

describe('stored chats', () => {
  it('roundtrips stored drafts and drops legacy active chat fields', () => {
    const raw = JSON.stringify({
      version: 1,
      drafts: {'s:server-chat': 'saved\nmessage'},
      active: {s: 'server-chat'},
    });

    const parsed = parseStoredChats(raw);

    expect(parsed).toStrictEqual({drafts: {'s:server-chat': 'saved\nmessage'}});
  });

  it('returns empty drafts when stored JSON is corrupt or uses an unsupported version', () => {
    const unsupported = JSON.stringify({version: 2, drafts: {s: 'draft'}});

    expect(parseStoredChats('{invalid')).toStrictEqual({drafts: {}});
    expect(parseStoredChats(unsupported)).toStrictEqual({drafts: {}});
  });

  it('ignores non-string draft entries', () => {
    const raw = JSON.stringify({version: 1, drafts: {valid: 'ok', invalid: 42}});

    const parsed = parseStoredChats(raw);

    expect(parsed).toStrictEqual({drafts: {valid: 'ok'}});
  });

  it('truncates drafts exceeding MAX_DRAFT_LENGTH', () => {
    const raw = JSON.stringify({version: 1, drafts: {x: 'a'.repeat(MAX_DRAFT_LENGTH + 1)}});

    const parsed = parseStoredChats(raw);

    expect(parsed.drafts['x'].length).toBe(MAX_DRAFT_LENGTH);
  });

  it('prunes drafts to listed Session IDs and drops empty or legacy chat keys', () => {
    const stored = {
      drafts: {keep: 'draft', empty: '', gone: 'old', 'keep:c': 'old chat', 'keep:capture': 'old'},
    };

    expect(pruneStoredChats(stored, ['keep', 'empty'])).toStrictEqual({drafts: {keep: 'draft'}});
    expect(pruneStoredChats(stored, [])).toStrictEqual({drafts: {}});
  });
});
