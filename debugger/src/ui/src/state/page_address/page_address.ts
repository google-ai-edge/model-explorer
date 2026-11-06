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

import type {ChatSummary, Permission, SessionSummary} from '../../server/contracts/sessions';

/** Where the Debug reading is: a turn, and an analysis of one of its stages. */
export interface DebugPlace {
  turn: number | null;
  stage: 'prefill' | 'decode' | null;
  /** The server's analysis id (kv, final_logits, sampling, intermediate_tensors). */
  analysis: string | null;
}

/** The New chat form's place in an address, in place of a chat's number. */
export const NEW_CHAT = 'new';

/**
 * What the address bar says, and so where the page is:
 *
 *     /                                                the home page
 *     /sessions/{id}                                   the Session's default chat (shownChat())
 *     /sessions/{id}/new                               the New chat form
 *     /sessions/{id}/chats/{c}                         chat c, in Chat mode
 *     …/debug                                          Debug mode, of chat c or the default
 *     …/debug/turns/{n}                                turn n
 *     …/debug/turns/{n}/{stage}/{analysis}             that analysis of the stage open
 *
 * Any other path under a Session's names nothing of it: the Session's page says so (`unmatched`).
 */
export interface PageAddress extends DebugPlace {
  /** The Session; null on the home page. */
  session: string | null;
  /** The chat: its number, NEW_CHAT for the New chat form, or null for the Session's default. */
  chat: number | typeof NEW_CHAT | null;
  mode: 'Chat' | 'Debug';
  /** The rest of a path under the Session that names nothing of it, such as `chats/abc`; null for
   *  every other address. */
  unmatched: string | null;
}

export const NO_PLACE: DebugPlace = {turn: null, stage: null, analysis: null};
export const HOME: PageAddress = {
  session: null,
  chat: null,
  mode: 'Chat',
  unmatched: null,
  ...NO_PLACE,
};

/** A whole number from 1, as a path segment writes it. */
const NUMBER = /^[1-9]\d{0,8}$/;

/**
 * The address of a path's segments, decoded; null when it names no page. A path under a Session
 * that names nothing of it is the Session's page, `unmatched`. The router's route of a Session
 * (app/session_route.ts) reads its segments with it.
 */
export function parseSegments(parts: readonly string[]): PageAddress | null {
  if (!parts.length) {
    return HOME;
  }
  if (parts[0] !== 'sessions' || !parts[1]) {
    return null;
  }
  return sessionPlace(parts) ?? {...HOME, session: parts[1], unmatched: parts.slice(2).join('/')};
}

/** The place a path's segments name in their Session; null when they name none. */
function sessionPlace(parts: readonly string[]): PageAddress | null {
  const address: PageAddress = {...HOME, session: parts[1]};
  let at = 2;
  if (at === parts.length) {
    return address;
  }
  if (parts[at] === NEW_CHAT) {
    return at + 1 === parts.length ? {...address, chat: NEW_CHAT} : null;
  }
  if (parts[at] === 'chats') {
    if (!NUMBER.test(parts[at + 1] ?? '')) {
      return null;
    }
    address.chat = Number(parts[at + 1]);
    at += 2;
    if (at === parts.length) {
      return address;
    }
  }
  if (parts[at++] !== 'debug') {
    return null;
  }
  address.mode = 'Debug';
  if (at === parts.length) {
    return address;
  }
  if (parts[at] !== 'turns' || !NUMBER.test(parts[at + 1] ?? '')) {
    return null;
  }
  address.turn = Number(parts[at + 1]);
  at += 2;
  if (at === parts.length) {
    return address;
  }
  const [stage, analysis] = [parts[at], parts[at + 1]];
  if (
    at + 2 !== parts.length ||
    (stage !== 'prefill' && stage !== 'decode') ||
    !/^[a-z_]{1,64}$/.test(analysis)
  ) {
    return null;
  }
  return {...address, stage, analysis};
}

/** The address of a URL's path (its query and fragment aside); null when it names no page. */
export function parsePageAddress(url: string): PageAddress | null {
  const path = url.split(/[?#]/)[0];
  try {
    return parseSegments(
      path
        .split('/')
        .filter((part) => part !== '')
        .map(decodeURIComponent),
    );
  } catch {
    return null;
  }
}

/** The path of an address, the one parsePageAddress reads back. */
export function pageAddressPath(address: PageAddress): string {
  if (address.session === null) {
    return '/';
  }
  let path = `/sessions/${encodeURIComponent(address.session)}`;
  if (address.unmatched !== null) {
    return `${path}/${address.unmatched.split('/').map(encodeURIComponent).join('/')}`;
  }
  if (address.chat === NEW_CHAT) {
    return `${path}/${NEW_CHAT}`;
  }
  if (address.chat !== null) {
    path = `${path}/chats/${address.chat}`;
  }
  if (address.mode === 'Chat') {
    return path;
  }
  path = `${path}/debug`;
  if (address.turn === null) {
    return path;
  }
  path = `${path}/turns/${address.turn}`;
  return address.stage && address.analysis ? `${path}/${address.stage}/${address.analysis}` : path;
}

/**
 * What a Session's page shows at an address: one of its chats, the New chat form, or, for the
 * number of a chat the Session does not have, neither.
 */
export interface ShownChat {
  chat: ChatSummary | null;
  form: boolean;
}

/**
 * What a Session's page shows at an address. An address that names no chat shows the live one;
 * without one, the newest chat (the highest number); without any, the New chat form, which says
 * why no chat starts if none can.
 */
export function shownChat(
  record: Pick<SessionSummary, 'chats' | 'live_chat'>,
  address: Pick<PageAddress, 'chat'> & Partial<Pick<PageAddress, 'unmatched'>>,
): ShownChat {
  const chats = record.chats ?? [];
  if (address.unmatched) {
    return {chat: null, form: false};
  }
  if (address.chat === NEW_CHAT) {
    return {chat: null, form: true};
  }
  if (address.chat !== null) {
    return {chat: chats.find((chat) => chat.n === address.chat) ?? null, form: false};
  }
  const live = chats.find((chat) => chat.n === record.live_chat);
  if (live) {
    return {chat: live, form: false};
  }
  const newest = chats.reduce<ChatSummary | null>(
    (found, chat) => (found && found.n > chat.n ? found : chat),
    null,
  );
  return newest ? {chat: newest, form: false} : {chat: null, form: true};
}

/** The mode a chat is read in, as the server allows it (its `debug` action): the one wanted,
 *  unless it refuses it and allows the other. A chat that exists is always read in Chat; one with
 *  no turn kept is read in Chat only. */
export function allowedMode(
  modes: {chat: Permission; debug: Permission},
  wanted: PageAddress['mode'],
): PageAddress['mode'] {
  const allows = (mode: PageAddress['mode']) =>
    (mode === 'Chat' ? modes.chat : modes.debug).allowed;
  const other = wanted === 'Chat' ? 'Debug' : 'Chat';
  return !allows(wanted) && allows(other) ? other : wanted;
}

export function samePlace(a: DebugPlace, b: DebugPlace): boolean {
  return a.turn === b.turn && a.stage === b.stage && a.analysis === b.analysis;
}
