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

import {DestroyRef, Injectable, signal, inject, computed, effect, untracked} from '@angular/core';
import {NavigationEnd, Router} from '@angular/router';
import {
  HOME,
  NEW_CHAT,
  NO_PLACE,
  allowedMode,
  pageAddressPath,
  parsePageAddress,
  shownChat,
  type DebugPlace,
  type PageAddress,
  type ShownChat,
} from './page_address/page_address';
import {captureId} from '../server/capture_id';
import {JobsService} from '../server/jobs_service';
import {RuntimeService} from '../server/runtime_service';
import {SessionListService} from '../server/session_list_service';
import {CaptureStore} from '../server/capture_store';
import type {Permission} from '../server/contracts/sessions';
import {emptyChats, parseStoredChats, pruneStoredChats, MAX_DRAFT_LENGTH} from './chat_storage';

/** A mode of a Session the list does not have (yet). */
const UNLISTED: Permission = {allowed: false, reason: ''};
/** Why the New chat form is not read in Debug. */
const NO_CHAT: Permission = {
  allowed: false,
  reason: 'Start a chat, or choose one, to inspect it in Debug.',
};
const READABLE: Permission = {allowed: true, reason: ''};
/** What a page that names no Session shows. */
const NOTHING: ShownChat = {chat: null, form: false};

/**
 * Where the page is, and what is attached there. The page's address (page_address/) is the one
 * navigation state: the Session, its chat and the mode are read from it, and going somewhere is a
 * navigation to another address. The Session's runtime, and the capture of the chat shown, are
 * attached from here. Which modes a chat is read in is the server's (its `debug` action; a chat is
 * always read in Chat): an address of a mode it refuses is replaced by the other one's, however
 * the page arrived there.
 */
@Injectable({providedIn: 'root'})
export class WorkspaceStateService {
  private readonly report = inject(CaptureStore);
  readonly runtime = inject(RuntimeService);
  readonly jobs = inject(JobsService);
  readonly sessions = inject(SessionListService);
  private readonly router = inject(Router);
  /** The page's address, as the router last arrived at it. */
  readonly address = signal<PageAddress>(HOME);
  readonly home = computed(() => this.address().session === null);
  readonly selectedId = computed(() => this.address().session);
  /** The mode the page is read in: the address's, or, while an address of a mode the server
   *  refuses is being replaced, the one it is replaced by. */
  readonly mode = computed(() => {
    const wanted = this.address().mode;
    return this.session()
      ? allowedMode({chat: this.modes().Chat, debug: this.modes().Debug}, wanted)
      : wanted;
  });
  readonly sessionName = computed(
    () =>
      this.sessions.items().find((item) => item.id === this.selectedId())?.name ??
      this.sessions.items()[0]?.name ??
      'Saved session',
  );
  private readonly restored = (() => {
    try {
      return parseStoredChats(localStorage.getItem('debugger.chats.v1'));
    } catch {
      return emptyChats();
    }
  })();
  readonly storageError = signal('');
  /** The Session the address names, as the server lists it; null on the home page, while the list
   *  loads, and for a Session the workspace does not have. */
  readonly session = computed(
    () => this.sessions.items().find((record) => record.id === this.selectedId()) ?? null,
  );
  /** The Session's record id, whose runtime RuntimeService is attached to from here only. */
  readonly captureRecordId = computed(() => this.session()?.id ?? null);
  /** What the page shows of the Session: one of its chats, or the New chat form. */
  readonly shown = computed(() => {
    const record = this.session();
    return record ? shownChat(record, this.address()) : NOTHING;
  });
  /** The chat shown, as the server lists it; null on the New chat form. */
  readonly chat = computed(() => this.shown().chat);
  /** Whether the page shows the New chat form. */
  readonly newChat = computed(() => this.shown().form);
  /**
   * The capture of the chat shown (capture_id.ts); CaptureStore is attached to it from here
   * only.
   */
  readonly captureId = computed(() => {
    const record = this.session();
    const chat = this.chat();
    return record && chat ? captureId(record.id, chat.n) : null;
  });
  private attachedRecordId: string | null = null;
  private attachedCaptureId: string | null = null;
  private attachRuntime(id: string | null): void {
    if (id === this.attachedRecordId) {
      return;
    }
    this.attachedRecordId = id;
    // A missing Session cannot borrow another Session's runtime.
    void this.runtime.attach(id);
  }
  private attachCapture(id: string | null): void {
    if (id === this.attachedCaptureId) {
      return;
    }
    this.attachedCaptureId = id;
    this.report.attach(id);
    if (this.chat()?.has_capture) {
      void this.report.load();
    }
  }
  readonly chatDrafts = signal(this.restored.drafts);
  private readonly draftKey = computed(() => this.selectedId() ?? '');
  readonly chatDraft = computed(() => this.chatDrafts()[this.draftKey()] ?? '');
  setChatDraft(value: string): void {
    if (this.selectedId()) {
      this.chatDrafts.update((d) => ({...d, [this.draftKey()]: value.slice(0, MAX_DRAFT_LENGTH)}));
    }
  }
  /** Whether the chat shown kept a turn. */
  readonly chatHasCapture = computed(() => !!this.chat()?.has_capture);
  /** Whether the server lets the chat shown be read in each mode now, and if not why. */
  readonly modes = computed(() => {
    if (!this.session()) {
      return {Chat: UNLISTED, Debug: UNLISTED};
    }
    const chat = this.chat();
    return {Chat: READABLE, Debug: chat ? chat.actions.debug : NO_CHAT};
  });
  /** Goes to a Session: its live chat, or what shownChat() shows without one. */
  open(id: string): void {
    this.navigate({...HOME, session: id});
  }
  /** Goes to chat `n` of the Session: in the mode the page is in, if the chat is read in it. */
  showChat(n: number): void {
    const record = this.session();
    if (!record) {
      return;
    }
    const chat = record.chats.find((item) => item.n === n);
    const mode = this.mode() === 'Debug' && chat?.actions.debug.allowed ? 'Debug' : 'Chat';
    this.navigate({...HOME, session: record.id, chat: n, mode});
  }
  /** Goes to the Session's New chat form. */
  showNewChat(): void {
    const id = this.selectedId();
    if (id !== null) {
      this.navigate({...HOME, session: id, chat: NEW_CHAT});
    }
  }
  /** The address, naming the chat shown: a mode or a Debug place it goes to stays with that chat
   *  whatever the Session's default becomes. */
  private pinned(): PageAddress {
    const address = this.address();
    const chat = this.chat();
    return address.chat === null && chat ? {...address, chat: chat.n} : address;
  }
  goHome(): void {
    this.navigate(HOME);
  }
  /** Where the conversation's Debug reading is, which Debug mode's address names; its page sets
   *  it while it is there. */
  readonly debugPlace = signal<DebugPlace>(NO_PLACE);
  /** Goes to the chat's Chat or Debug mode, one the server allows; Debug returns to where its
   *  reading is. */
  setMode(mode: 'Chat' | 'Debug', replace = false): void {
    const address = this.pinned();
    if (address.session === null || (this.session() && !this.modes()[mode].allowed)) {
      return;
    }
    this.navigate(
      mode === 'Debug' ? {...address, mode, ...this.debugPlace()} : {...address, mode, ...NO_PLACE},
      replace,
    );
  }
  /** Writes in the address where the Debug reading is now. */
  showDebugPlace(place: DebugPlace, replace = false): void {
    const address = this.pinned();
    if (address.session !== null && address.mode === 'Debug') {
      this.navigate({...address, ...place}, replace);
    }
  }
  /** The address of a place of the chat's Debug reading. */
  debugAddress(place: DebugPlace): string {
    return pageAddressPath({...this.pinned(), mode: 'Debug', ...place});
  }
  private navigate(address: PageAddress, replace = false): void {
    this.navigateTo(pageAddressPath(address), replace);
  }
  /** Goes to a path, keeping the query (?debugPalette, say) the page was opened with. */
  private navigateTo(path: string, replace = false): void {
    const url = this.router.url;
    const query = url.includes('?') ? url.slice(url.indexOf('?')) : '';
    void this.router.navigateByUrl(`${path}${query}`, {replaceUrl: replace});
  }
  private readonly completedDrafts = new Set<string>();
  /** The chat starts whose chat was shown. */
  private readonly shownChats = new Set<string>();
  private pendingStorage: object | null = null;
  private storageTimer: ReturnType<typeof setTimeout> | null = null;
  private pruned = false;
  private flushStorage(): void {
    if (this.storageTimer !== null) {
      clearTimeout(this.storageTimer);
    }
    this.storageTimer = null;
    const data = this.pendingStorage;
    if (!data) {
      return;
    }
    this.pendingStorage = null;
    try {
      localStorage.setItem('debugger.chats.v1', JSON.stringify(data));
      this.storageError.set('');
    } catch {
      this.storageError.set('Draft could not be saved in this browser. Keep this page open.');
    }
  }
  constructor() {
    const navigation = this.router.events.subscribe((event) => {
      if (event instanceof NavigationEnd) {
        this.address.set(parsePageAddress(event.urlAfterRedirects) ?? HOME);
      }
    });
    const onPageHide = () => this.flushStorage();
    if (typeof window !== 'undefined') {
      window.addEventListener('pagehide', onPageHide);
    }
    inject(DestroyRef).onDestroy(() => {
      navigation.unsubscribe();
      if (typeof window !== 'undefined') {
        window.removeEventListener('pagehide', onPageHide);
      }
      this.flushStorage();
    });
    // An address of a mode the server refuses (typed, a link, Back or Forward, or one the Session
    // ended at) is replaced by the address of the mode it is read in.
    effect(() => {
      const mode = this.mode();
      if (this.session() && mode !== this.address().mode) {
        untracked(() => this.setMode(mode, true));
      }
    });
    // Navigation only changes the address; the attachments follow the resolved context: the
    // runtime the Session's, the capture the chat shown's.
    effect(() => {
      const id = this.captureRecordId();
      untracked(() => this.attachRuntime(id));
    });
    effect(() => {
      const id = this.captureId();
      untracked(() => this.attachCapture(id));
    });
    // The Session's own address (/sessions/{id}) is replaced, once the list resolves it, by the
    // address of what it shows: that chat, or the New chat form. What the page shows then stays
    // put as chats start and end: a Stop or a failed turn leaves the chat shown, and the form,
    // with what was typed in it, stays while its chat starts and after the start failed.
    effect(() => {
      const address = this.address();
      const shown = this.shown();
      if (
        !this.session() ||
        address.session === null ||
        address.chat !== null ||
        address.unmatched !== null
      ) {
        return;
      }
      const chat = shown.chat ? shown.chat.n : shown.form ? NEW_CHAT : null;
      if (chat !== null) {
        untracked(() => this.navigate({...address, chat}, true));
      }
    });
    // A chat this page started is shown once the list has it, unless the reader went elsewhere
    // meanwhile: the New chat form, or the Session's default, then shows it.
    effect(() => {
      const job = this.jobs.job();
      const record = this.session();
      const chatNumber = job?.chat;
      if (
        job?.operation !== 'chat' ||
        job.status !== 'completed' ||
        chatNumber == null ||
        job.session_id !== record?.id ||
        this.shownChats.has(job.id) ||
        !record.chats.some((chat) => chat.n === chatNumber)
      ) {
        return;
      }
      this.shownChats.add(job.id);
      const address = untracked(() => this.address());
      if (address.chat === NEW_CHAT || address.chat === null) {
        untracked(() => this.navigate({...HOME, session: record.id, chat: chatNumber}, true));
      }
    });
    // A job another tab started for this Session, which the polled list names, is followed as a
    // reload follows it.
    effect(() => {
      const jobId = this.session()?.job_id;
      untracked(() => void this.runtime.follow(jobId));
    });
    effect(() => {
      const job = this.jobs.job();
      if (
        job?.operation === 'generate' &&
        job.status === 'completed' &&
        job.session_id === this.captureRecordId() &&
        !this.completedDrafts.has(job.id)
      ) {
        this.completedDrafts.add(job.id);
        if (job.prompt && this.chatDraft() === job.prompt) {
          this.setChatDraft('');
        }
      }
    });
    // Drafts change on every keystroke; write them at most every 300 ms and on page hide.
    effect(() => {
      const data = {version: 1, drafts: this.chatDrafts()};
      this.pendingStorage = data;
      if (this.storageTimer !== null) {
        clearTimeout(this.storageTimer);
      }
      this.storageTimer = setTimeout(() => this.flushStorage(), 300);
    });
    // Drop drafts for Sessions the server no longer lists.
    effect(() => {
      const listing = this.sessions.listing();
      if (!listing || this.pruned) {
        return;
      }
      this.pruned = true;
      const known = listing.sessions.map((session) => session.id);
      untracked(() =>
        this.chatDrafts.set(pruneStoredChats({drafts: this.chatDrafts()}, known).drafts),
      );
    });
  }
}
