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

import {A11yModule} from '@angular/cdk/a11y';
import {OverlayModule} from '@angular/cdk/overlay';
import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  inject,
  signal,
  viewChild,
  viewChildren,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import {
  ConversationBookmark,
  WorkspaceStateService,
} from '../../../app/workspace_state_service';
import {
  OverlayDialog,
  OverlayPanel,
} from '../../../shared/overlay_dialog/overlay_dialog';
import {CONVERSATION_BOOKMARK_CODEC} from '../conversation_bookmarks';
/** Demo bookmark rows on the existing CDK overlay and Material controls. */
@Component({
  selector: 'conversation-bookmarks',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    OverlayDialog,
    OverlayPanel,
    OverlayModule,
    A11yModule,
    MatButtonModule,
    MatIconModule,
    MatTooltipModule,
  ],
  template: ` <button
      mat-button
      class="bookmark-select"
      aria-label="Bookmarks"
      matTooltip="Bookmarks"
      aria-haspopup="dialog"
      [attr.aria-expanded]="open()"
      cdkOverlayOrigin
      #origin="cdkOverlayOrigin"
      (click)="open.set(!open())"
      (keydown.arrowdown)="$event.preventDefault(); open.set(true)"
    >
      <mat-icon>bookmarks</mat-icon><span>Bookmarks</span>
    </button>
    <ng-template
      overlayDialog
      [origin]="origin"
      [open]="open()"
      (closed)="open.set(false)"
      (overlayKeydown)="key($event)"
    >
      <section
        class="bookmark-panel"
        [class.empty]="!workspace.bookmarks().length"
        overlayPanel
        aria-label="Chat bookmarks"
      >
        @for (bookmark of workspace.bookmarks(); track keyFor(bookmark)) {
          <div class="bookmark-row" [attr.data-bookmark-key]="keyFor(bookmark)">
            <button
              #bookmarkLocation
              mat-button
              class="bookmark-location"
              [attr.aria-label]="'Open bookmark ' + label(bookmark)"
              [matTooltip]="label(bookmark)"
              (click)="select(bookmark)"
            >
              <span class="bookmark-title">{{ label(bookmark) }}</span
              ><span class="bookmark-view"
                >Token Diff · {{ bookmark.alignment === 'content' ? 'Context' : 'Steps' }}</span
              >
            </button>
            <button
              mat-icon-button
              aria-label="Move bookmark up"
              matTooltip="Move up"
              [disabled]="$first"
              (click)="move($index, -1, $event)"
            >
              <mat-icon>keyboard_arrow_up</mat-icon>
            </button>
            <button
              mat-icon-button
              aria-label="Move bookmark down"
              matTooltip="Move down"
              [disabled]="$last"
              (click)="move($index, 1, $event)"
            >
              <mat-icon>keyboard_arrow_down</mat-icon>
            </button>
            <button
              mat-icon-button
              aria-label="Delete bookmark"
              matTooltip="Delete bookmark"
              (click)="remove($index, $event)"
            >
              <mat-icon>delete_outline</mat-icon>
            </button>
          </div>
        } @empty {
          <div #bookmarkEmpty class="bookmark-empty" tabindex="0" cdkFocusInitial>
            <strong>No bookmarks yet</strong
            ><span>Select a stage or token, then add a bookmark.</span>
          </div>
        }
        <span class="sr-only" role="status">{{ notice() }}</span>
      </section>
    </ng-template>`,
  styleUrl: './conversation_bookmarks.scss',
})
export class ConversationBookmarks {
  readonly workspace = inject(WorkspaceStateService);
  readonly open = signal(false);
  readonly notice = signal('');
  private readonly locationButtons =
    viewChildren<ElementRef<HTMLButtonElement>>('bookmarkLocation');
  private readonly emptyNotice =
    viewChild<ElementRef<HTMLElement>>('bookmarkEmpty');
  keyFor(b: ConversationBookmark) {
    return CONVERSATION_BOOKMARK_CODEC.identity(b);
  }
  label(b: ConversationBookmark) {
    return b.title ?? `Turn ${b.turn} · Aligned position ${b.index}`;
  }
  select(b: ConversationBookmark) {
    this.open.set(false);
    this.workspace.openBookmark(b);
  }
  move(index: number, delta: number, _event: Event) {
    const list = [...this.workspace.bookmarks()];
    const next = index + delta;
    if (next < 0 || next >= list.length) {
      return;
    }
    const [item] = list.splice(index, 1);
    list.splice(next, 0, item);
    this.workspace.setBookmarks(list);
    this.notice.set('Bookmark moved to position ' + (next + 1));
    requestAnimationFrame(() => {
      this.locationButtons()[next]?.nativeElement.focus();
    });
  }
  remove(index: number, _event: Event) {
    const list = [...this.workspace.bookmarks()];
    list.splice(index, 1);
    this.workspace.setBookmarks(list);
    this.notice.set('Bookmark deleted');
    requestAnimationFrame(() => {
      const rows = this.locationButtons().map((item) => item.nativeElement);
      if (rows.length) {
        rows[Math.min(index, rows.length - 1)].focus();
      } else {
        this.emptyNotice()?.nativeElement.focus();
      }
    });
  }
  key(event: KeyboardEvent) {
    if (!['ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) {
      return;
    }
    const rows = this.locationButtons().map((item) => item.nativeElement);
    if (!rows.length) {
      return;
    }
    event.preventDefault();
    const target = event.target as HTMLElement;
    const activeIndex = rows.findIndex(
      (btn) => btn === target || btn.closest('.bookmark-row')?.contains(target),
    );
    const at = activeIndex >= 0 ? activeIndex : 0;
    rows[
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? rows.length - 1
          : Math.max(
              0,
              Math.min(
                rows.length - 1,
                at + (event.key === 'ArrowUp' ? -1 : 1),
              ),
            )
    ].focus();
  }
}
