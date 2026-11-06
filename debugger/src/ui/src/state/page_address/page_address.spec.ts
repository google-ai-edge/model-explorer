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

import {allowed, chat, refused, session as record} from '../../server/testing/session_fixtures';
import {
  HOME,
  NEW_CHAT,
  PageAddress,
  allowedMode,
  pageAddressPath,
  parsePageAddress,
  shownChat,
} from './page_address';

const session: PageAddress = {...HOME, session: 's1'};
const ADDRESSES: [string, PageAddress][] = [
  ['/', HOME],
  ['/sessions/s1', session],
  ['/sessions/s1/new', {...session, chat: NEW_CHAT}],
  ['/sessions/s1/chats/2', {...session, chat: 2}],
  ['/sessions/s1/chats/2/debug', {...session, chat: 2, mode: 'Debug'}],
  [
    '/sessions/s1/chats/12/debug/turns/3/decode/kv',
    {...session, chat: 12, mode: 'Debug', turn: 3, stage: 'decode', analysis: 'kv'},
  ],
  ['/sessions/s1/debug', {...session, mode: 'Debug'}],
  ['/sessions/s1/debug/turns/3', {...session, mode: 'Debug', turn: 3}],
  [
    '/sessions/s1/debug/turns/12/decode/final_logits',
    {...session, mode: 'Debug', turn: 12, stage: 'decode', analysis: 'final_logits'},
  ],
  [
    '/sessions/s1/debug/turns/1/prefill/kv',
    {...session, mode: 'Debug', turn: 1, stage: 'prefill', analysis: 'kv'},
  ],
];

describe('page addresses', () => {
  it('each kind of address is read, and written back the same', () => {
    for (const [path, address] of ADDRESSES) {
      expect(parsePageAddress(path), path).toStrictEqual(address);
      expect(pageAddressPath(address), path).toBe(path);
    }
  });

  it('a query, a fragment, a trailing slash and escapes do not change the page', () => {
    expect(parsePageAddress('/sessions/s1/debug?debugPalette=teal#x')).toStrictEqual({
      ...session,
      mode: 'Debug',
    });
    expect(parsePageAddress('/sessions/s1/')).toStrictEqual(session);
    const odd = {...session, session: 'a b/c'};
    expect(pageAddressPath(odd)).toBe('/sessions/a%20b%2Fc');
    expect(parsePageAddress(pageAddressPath(odd))).toStrictEqual(odd);
  });

  it('a path that names no page is no address', () => {
    for (const path of ['/sessions', '/session/s1', '/sessions/%E0%A4%A']) {
      expect(parsePageAddress(path), path).toBe(null);
    }
  });

  it("a path under a Session that names nothing of it is the Session's, unmatched", () => {
    for (const rest of [
      'chat',
      'chats',
      'chats/c2',
      'chats/abc/debug',
      'chats/0',
      'chats/2/new',
      'new/debug',
      'new/2',
      'debug/chats/2',
      'debug/turns',
      'debug/turns/0',
      'debug/turns/-1',
      'debug/turns/1.5',
      'debug/turns/2/decode',
      'debug/turns/2/middle/kv',
      'debug/turns/2/decode/KV',
      'debug/turns/2/decode/kv/more',
      'turns/2',
    ]) {
      const path = `/sessions/s1/${rest}`;
      const address = {...session, unmatched: rest};
      expect(parsePageAddress(path), path).toStrictEqual(address);
      expect(pageAddressPath(address), path).toBe(path);
    }
    // It shows no chat, nor the form.
    const listed = record('s1', {chats: [chat(1)], live_chat: 1});
    expect(shownChat(listed, {chat: null, unmatched: 'chats/abc'})).toStrictEqual({
      chat: null,
      form: false,
    });
  });

  it('an address that names no chat shows the live one, else the newest, else the New chat form', () => {
    const first = chat(1, {state: 'ended', has_capture: true});
    const second = chat(2, {state: 'ended'});
    const live = chat(3);
    const startable = {...record().actions, new_chat: allowed};
    const on = record('s1', {chats: [first, second, live], live_chat: 3, actions: startable});
    const after = record('s1', {chats: [first, second], live_chat: null, actions: startable});
    const ended = record('s1', {
      chats: [first, second],
      live_chat: null,
      actions: {...record().actions, new_chat: refused('This Session has ended.')},
    });
    const fresh = record('s1', {chats: [], live_chat: null, actions: startable});
    const starting = record('s1', {chats: [], live_chat: null});
    const at = (chatValue: PageAddress['chat']) => ({chat: chatValue});
    expect(shownChat(on, at(null))).toStrictEqual({chat: live, form: false});
    expect(shownChat(on, at(1))).toStrictEqual({chat: first, form: false});
    expect(shownChat(on, at(NEW_CHAT))).toStrictEqual({chat: null, form: true});
    expect(shownChat(on, at(9)), 'a chat the Session lacks').toStrictEqual({
      chat: null,
      form: false,
    });
    // The newest chat, whether it kept a turn or not, in either mode.
    expect(shownChat(after, at(null)), 'the live chat ended').toStrictEqual({
      chat: second,
      form: false,
    });
    const debugAddress: PageAddress = {...session, mode: 'Debug'};
    expect(shownChat(after, debugAddress)).toStrictEqual({chat: second, form: false});
    expect(shownChat(ended, at(null))).toStrictEqual({chat: second, form: false});
    const unordered = {...ended, chats: [second, first]};
    expect(shownChat(unordered, at(null)), 'by number').toStrictEqual({chat: second, form: false});
    expect(shownChat(fresh, at(null))).toStrictEqual({chat: null, form: true});
    // No chat starts yet, and none was: the form says why.
    expect(shownChat(starting, at(null))).toStrictEqual({chat: null, form: true});
  });

  it('a chat is read in the mode wanted, unless the server allows only the other', () => {
    const ended = {chat: refused('This Session has ended.'), debug: allowed};
    const uncaptured = {chat: allowed, debug: refused('This chat has no turn to inspect.')};
    const both = {chat: allowed, debug: allowed};
    const neither = {chat: refused('x'), debug: refused('y')};
    for (const wanted of ['Chat', 'Debug'] as const) {
      expect(allowedMode(ended, wanted), wanted).toBe('Debug');
      expect(allowedMode(uncaptured, wanted), wanted).toBe('Chat');
      expect(allowedMode(both, wanted), wanted).toBe(wanted);
      expect(allowedMode(neither, wanted), wanted).toBe(wanted);
    }
  });
});
