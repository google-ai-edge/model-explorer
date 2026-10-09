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

import {
  computed,
  effect,
  inject,
  Injectable,
  signal,
  untracked,
} from '@angular/core';
import {ReportStateService} from '../data/report_state_service';
import {RuntimeService} from '../data/runtime_service';
import {SessionListService} from '../data/session_list_service';
import {
  CONVERSATION_BOOKMARK_CODEC,
  ConversationBookmark,
} from '../features/conversation/conversation_bookmarks';
import type {
  ConversationReadingPosition,
  TokenDiffViewState,
} from '../features/conversation/conversation_view_state';
import {
  emptyChats,
  MAX_DRAFT_LENGTH,
  parseStoredChats,
  pruneStoredChats,
} from '../features/sessions/chat_storage';
import {BookmarkStore} from '../shared/bookmark_store/bookmark_store';
import {resolveWorkspaceContext} from './workspace_context';
export type {ConversationBookmark} from '../features/conversation/conversation_bookmarks';
export type DebugView = 'Token Diff' | 'Graph Diff' | 'KV Diff';
/** Navigation is independent of the shared capture selection. */
@Injectable({providedIn: 'root'})
export class WorkspaceStateService {
  readonly report = inject(ReportStateService);
  readonly runtime = inject(RuntimeService);
  readonly sessions = inject(SessionListService);
  readonly selectedId = signal<string | null>(null);
  readonly sessionName = computed(
    () =>
      this.sessions.items().find((item) => item.id === this.selectedId())
        ?.name ??
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
  private readonly activeChats = signal(this.restored.active);
  readonly activeChat = signal('capture');
  readonly context = computed(() =>
    resolveWorkspaceContext(
      this.selectedId(),
      this.activeChat(),
      this.sessions.allItems(),
    ),
  );
  /** The one capture identity; ReportStateService and RuntimeService are attached from here only. */
  readonly captureRecordId = computed(() => this.context().captureRecordId);
  private attachedCaptureId: string | null = null;
  private attach(id: string | null) {
    if (id === this.attachedCaptureId) {
      return;
    }
    this.attachedCaptureId = id;
    this.report.attach(id);
    if (this.context().activeRecord?.has_capture) {
      void this.report.load();
    }
    // Missing and legacy local drafts cannot borrow another Chat's runtime.
    void this.runtime.attach(id);
  }
  private navigationRevision = 0;
  readonly localChats = signal(this.restored.chats);
  readonly chats = computed(() => [
    {id: 'capture', name: 'Captured conversation'},
    ...this.sessions
      .allItems()
      .filter((c) => c.parent_session_id === this.selectedId())
      .map((c) => ({id: c.id, name: c.name})),
    ...(this.localChats()[this.selectedId() ?? ''] ?? []),
  ]);
  readonly chatName = computed(
    () =>
      this.chats().find((c) => c.id === this.activeChat())?.name ??
      'Captured conversation',
  );
  readonly chatDrafts = signal(this.restored.drafts);
  private readonly draftKey = computed(
    () => (this.selectedId() ?? '') + ':' + this.activeChat(),
  );
  readonly chatDraft = computed(() => this.chatDrafts()[this.draftKey()] ?? '');
  setChatDraft(value: string) {
    if (this.selectedId()) {
      this.chatDrafts.update((d) => ({
        ...d,
        [this.draftKey()]: value.slice(0, MAX_DRAFT_LENGTH),
      }));
    }
  }
  /** The server rejects a new Chat while another one is active or pending. */
  readonly newChatBlocked = computed(
    () => this.runtime.execution()?.newChatAllowed === false,
  );
  readonly creatingChat = signal(false);
  readonly chatError = signal('');
  readonly chatHasCapture = computed(
    () => !!this.runtime.selected()?.has_capture,
  );
  async newChat() {
    const session = this.selectedId();
    if (!session || this.creatingChat()) {
      return;
    }
    const navigation = ++this.navigationRevision;
    this.creatingChat.set(true);
    this.chatError.set('');
    try {
      const chat = await this.sessions.manage('chat', {
        id: session,
        name: 'New chat ' + this.chats().length,
      });
      if (
        navigation === this.navigationRevision &&
        session === this.selectedId()
      ) {
        this.chooseChat(chat.id);
      }
    } catch (error) {
      if (navigation === this.navigationRevision) {
        this.chatError.set(
          error instanceof Error ? error.message : String(error),
        );
      }
    } finally {
      this.creatingChat.set(false);
    }
  }
  async chooseChat(id: string) {
    if (!this.chats().some((c) => c.id === id)) {
      return;
    }
    const navigation = ++this.navigationRevision;
    const parent = this.selectedId();
    const legacy = this.localChats()[parent ?? '']?.find((c) => c.id === id);
    if (parent && legacy) {
      this.activeChat.set(id);
      try {
        const created = await this.sessions.manage('chat', {
          id: parent,
          name: legacy.name,
        });
        this.chatDrafts.update((d) => ({
          ...d,
          [parent + ':' + created.id]: d[parent + ':' + id] ?? '',
        }));
        this.localChats.update((c) => ({
          ...c,
          [parent]: (c[parent] ?? []).filter((v) => v.id !== id),
        }));
        if (
          navigation !== this.navigationRevision ||
          parent !== this.selectedId()
        ) {
          return;
        }
        id = created.id;
      } catch (e) {
        if (navigation === this.navigationRevision) {
          this.chatError.set(
            'Could not connect this saved draft: ' + String(e),
          );
        }
        return;
      }
    }
    this.activeChat.set(id);
    const session = this.selectedId();
    if (session) {
      this.activeChats.update((c) => ({...c, [session]: id}));
    }
    this.mode.set('Chat');
  }
  private readonly completedDrafts = new Set<string>();
  private pendingStorage: object | null = null;
  private storageTimer: ReturnType<typeof setTimeout> | null = null;
  private pruned = false;
  private flushStorage() {
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
      this.storageError.set(
        'Draft could not be saved in this browser. Keep this page open.',
      );
    }
  }
  constructor() {
    // Navigation only changes signals; the capture attachment follows the resolved context.
    effect(() => {
      const id = this.captureRecordId();
      untracked(() => this.attach(id));
    });
    effect(() => {
      const job = this.runtime.job();
      if (
        job?.operation === 'generate' &&
        job.status === 'completed' &&
        job.session_id === this.context().captureRecordId &&
        !this.completedDrafts.has(job.id)
      ) {
        this.completedDrafts.add(job.id);
        if (
          job.prompt &&
          job.session_id === this.context().captureRecordId &&
          this.chatDraft() === job.prompt
        ) {
          this.setChatDraft('');
        }
      }
    });
    // Drafts change on every keystroke; write them at most every 300 ms and on page hide.
    effect(() => {
      const data = {
        version: 1,
        chats: this.localChats(),
        drafts: this.chatDrafts(),
        active: this.activeChats(),
      };
      this.pendingStorage = data;
      if (this.storageTimer !== null) {
        clearTimeout(this.storageTimer);
      }
      this.storageTimer = setTimeout(() => this.flushStorage(), 300);
    });
    if (typeof window !== 'undefined') {
      window.addEventListener('pagehide', () => this.flushStorage());
    }
    // Drop drafts and local Chats for Sessions the server no longer lists.
    effect(() => {
      const listing = this.sessions.listing();
      if (!listing || this.pruned) {
        return;
      }
      this.pruned = true;
      const known = listing.sessions.map((session) => session.id);
      untracked(() => {
        const pruned = pruneStoredChats(
          {
            chats: this.localChats(),
            drafts: this.chatDrafts(),
            active: this.activeChats(),
          },
          known,
        );
        this.localChats.set(pruned.chats);
        this.chatDrafts.set(pruned.drafts);
        this.activeChats.set(pruned.active);
      });
    });
  }
  readonly bookmarkTarget = signal<ConversationBookmark | null>(null);
  private readonly bookmarkStore = new BookmarkStore(
    CONVERSATION_BOOKMARK_CODEC,
  );
  /** Bookmarks belong to one Session, or to one Chat within it. */
  private readonly bookmarkKey = computed(
    () =>
      'debugger.bookmarks.' +
      this.selectedId() +
      (this.activeChat() === 'capture' ? '' : ':' + this.activeChat()),
  );
  readonly bookmarks = computed(() =>
    this.bookmarkStore.read(this.bookmarkKey()),
  );
  readonly bookmarkError = this.bookmarkStore.error;
  setBookmarks(items: ConversationBookmark[]) {
    this.bookmarkStore.set(this.bookmarkKey(), items);
  }
  openBookmark(bookmark: ConversationBookmark) {
    this.view.set('Token Diff');
    this.mode.set('Debug');
    this.bookmarkTarget.set({...bookmark});
  }
  readonly conversationViews = new Map<string, TokenDiffViewState>();
  readonly readingPositions = new Map<string, ConversationReadingPosition>();
  readonly home = signal(true);
  readonly mode = signal<'Chat' | 'Debug'>('Chat');
  readonly view = signal<DebugView>('Token Diff');
  open(id?: string) {
    this.navigationRevision++;
    this.activeChat.set('capture');
    this.mode.set('Chat');
    this.view.set('Token Diff');
    this.selectedId.set(id ?? this.sessions.items()[0]?.id ?? null);
    const active = this.activeChats()[this.selectedId() ?? ''] ?? 'capture';
    this.activeChat.set(
      this.chats().some((c) => c.id === active) ? active : 'capture',
    );
    this.home.set(false);
    if (
      this.localChats()[this.selectedId() ?? '']?.some(
        (c) => c.id === this.activeChat(),
      )
    ) {
      void this.chooseChat(this.activeChat());
    }
  }
  graph(turn: number, phase: string) {
    this.report.selectTurn(turn);
    this.report.selectPhase(phase);
    this.view.set('Graph Diff');
    this.mode.set('Debug');
    this.home.set(false);
  }
}
