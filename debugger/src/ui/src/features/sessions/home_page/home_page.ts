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

import {Clipboard} from '@angular/cdk/clipboard';
import {DatePipe} from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  inject,
  signal,
} from '@angular/core';
import {MatButtonModule, MatIconButton} from '@angular/material/button';
import {MatDialog} from '@angular/material/dialog';
import {MatIconModule} from '@angular/material/icon';
import {MatMenuModule} from '@angular/material/menu';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatSnackBar} from '@angular/material/snack-bar';
import {MatTooltipModule} from '@angular/material/tooltip';
import {PreferencesService} from '../../../app/preferences_service';
import {WorkspaceStateService} from '../../../app/workspace_state_service';
import {SessionSummary} from '../../../data/contracts/session_summary';
import {RuntimeService} from '../../../data/runtime_service';
import {SessionListService} from '../../../data/session_list_service';
import {
  sessionConfigurationLocked,
  sessionEnterable,
} from '../../../data/session_status';
import {ModelServerSwitch} from '../model_server_switch/model_server_switch';
import type {EditorRequest} from '../session_editor/session_editor';
import {SessionEditor} from '../session_editor/session_editor';
import {WelcomeCard} from '../welcome_card/welcome_card';
@Component({
  selector: 'home-page',
  imports: [
    SessionEditor,
    ModelServerSwitch,
    DatePipe,
    WelcomeCard,
    MatButtonModule,
    MatIconModule,
    MatTooltipModule,
    MatMenuModule,
    MatProgressSpinnerModule,
  ],
  templateUrl: './home_page.ng.html',
  styleUrl: './home_page.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class HomePage {
  reloadPage() {
    window.location.reload();
  }
  readonly editor = signal<EditorRequest | null>(null);
  private editorOrigin: {id: string; action: 'open' | 'more'} | null = null;
  newSession() {
    if (!this.sessions.listing()?.capabilities.create) return;
    this.editorOrigin = null;
    this.editor.set({mode: 'create'});
  }
  editSession(
    session: SessionSummary,
    mode: 'update' | 'duplicate' = 'update',
    action: 'open' | 'more' = 'more',
  ) {
    if (
      !this.sessions.listing()?.capabilities[
        mode === 'duplicate' ? 'duplicate' : 'create'
      ] ||
      sessionConfigurationLocked(session) ||
      (mode === 'update' && session.has_capture)
    )
      return;
    this.editorOrigin = {id: session.id, action};
    this.editor.set({session, mode});
  }
  /** A created Session stays in the list while its Model Server starts; Open unlocks once it is on. */
  sessionCreated() {
    this.snackBar.open(
      'Session created. Its Model Server is starting.',
      'Dismiss',
      {
        duration: 5000,
      },
    );
  }
  finishEditor(saved: boolean) {
    const origin = this.editorOrigin,
      updated = this.editor()?.mode === 'update';
    this.editor.set(null);
    // The list is recreated after editing, so resolve the new trigger by session ID.
    requestAnimationFrame(() => {
      const row = origin
        ? Array.from(
            this.element.nativeElement.querySelectorAll<HTMLElement>(
              '[data-session-id]',
            ),
          ).find((el) => el.dataset['sessionId'] === origin.id)
        : null;
      const trigger = row?.querySelector<HTMLButtonElement>(
        `[data-session-action="${origin?.action}"]`,
      );
      (
        trigger ??
        this.element.nativeElement.querySelector<HTMLButtonElement>(
          '.new-session',
        )
      )?.focus();
    });
    if (saved && (updated || !this.sessions.listing()?.generation_available))
      this.snackBar.open('Session configuration saved', 'Dismiss', {
        duration: 3000,
      });
  }
  openSession(session: SessionSummary) {
    if (
      session.has_capture === false &&
      !this.sessions.listing()?.generation_available
    )
      this.editSession(session, 'update', 'open');
    else this.workspace.open(session.id);
  }
  async deleteSession(session: SessionSummary, trigger: MatIconButton) {
    try {
      const {SessionDelete} = await import('../session_delete/session_delete');
      this.dialogs
        .open(SessionDelete, {data: session, width: '466px'})
        .afterClosed()
        .subscribe((deleted) => {
          if (!deleted) trigger.focus();
          else
            requestAnimationFrame(() =>
              this.element.nativeElement
                .querySelector<HTMLButtonElement>('.new-session')
                ?.focus(),
            );
          if (deleted)
            this.snackBar
              .open('Session deleted', 'Undo', {duration: 8000})
              .onAction()
              .subscribe(() => {
                void this.sessions
                  .manage('restore', {id: session.id})
                  .catch((e) => this.snackBar.open(e.message, 'Dismiss'));
              });
        });
    } catch {
      this.dialogError('Delete dialog');
      trigger.focus();
    }
  }
  private readonly element: ElementRef<HTMLElement> = inject(ElementRef);
  readonly editing = signal<string | null>(null);
  readonly saving = signal(false);
  /** The one way into a Session. It needs something to do there: a Model Server that is on, or
   *  captured data to read. Without a runtime, an empty Session opens its configuration instead. */
  canOpen(session: SessionSummary) {
    return this.sessions.listing()?.generation_available
      ? sessionEnterable(session)
      : session.has_capture !== false ||
          !!this.sessions.listing()?.capabilities?.create;
  }
  openHint(session: SessionSummary) {
    return this.canOpen(session)
      ? 'Open session'
      : this.runtime.modelServerState(session) === 'starting'
        ? 'The Model Server is starting'
        : 'Turn the Model Server on to enter this Session';
  }
  readonly renameError = signal('');
  private renameTrigger: MatIconButton | null = null;
  startRename(session: SessionSummary, trigger: MatIconButton) {
    if (this.saving()) return;
    this.renameTrigger = trigger;
    this.editing.set(session.id);
    this.renameError.set('');
    requestAnimationFrame(() => {
      const input = this.element.nativeElement.querySelector(
        '.session-name-input',
      ) as HTMLInputElement | null;
      input?.focus();
      input?.select();
    });
  }
  async finishRename(
    session: SessionSummary,
    input: HTMLInputElement,
    save: boolean,
    restoreFocus = false,
  ) {
    if (this.editing() !== session.id || this.saving()) return;
    const trigger = this.renameTrigger;
    if (!save) {
      this.editing.set(null);
      requestAnimationFrame(() => trigger?.focus());
      return;
    }
    const name = input.value.trim();
    if (!name) {
      if (restoreFocus) {
        input.setCustomValidity('Enter a session name');
        input.reportValidity();
      } else this.editing.set(null);
      return;
    }
    this.saving.set(true);
    this.renameError.set('');
    try {
      if (name !== session.name) await this.sessions.rename(session.id, name);
      if (this.editing() !== session.id) return;
      const document = this.element.nativeElement.ownerDocument;
      // Keep a completed blur save from dropping focus when its input disappears.
      const returnFocus =
        input.isConnected &&
        (restoreFocus ||
          document.activeElement === input ||
          document.activeElement === document.body);
      this.editing.set(null);
      if (returnFocus) requestAnimationFrame(() => trigger?.focus());
    } catch (error) {
      if (this.editing() === session.id) {
        this.renameError.set((error as Error).message);
        if (input.isConnected) input.focus();
      }
    } finally {
      this.saving.set(false);
    }
  }
  renameKey(
    event: KeyboardEvent,
    session: SessionSummary,
    input: HTMLInputElement,
  ) {
    if (event.isComposing) return;
    if (event.key === 'Enter' || event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      void this.finishRename(session, input, event.key === 'Enter', true);
    }
  }
  readonly preferences = inject(PreferencesService);
  readonly sessions = inject(SessionListService);
  readonly runtime = inject(RuntimeService);
  readonly workspace = inject(WorkspaceStateService);
  private readonly clipboard = inject(Clipboard);
  private readonly dialogs = inject(MatDialog);
  private readonly snackBar = inject(MatSnackBar);
  copy(session: SessionSummary) {
    const copied = this.clipboard.copy(session.id);
    this.snackBar.open(
      copied ? 'Session ID copied' : 'Could not copy session ID',
      'Dismiss',
      {
        duration: 3000,
      },
    );
  }
  private dialogError(label: string) {
    this.snackBar
      .open(
        label + ' could not be loaded. Reload the page and try again.',
        'Reload',
      )
      .onAction()
      .subscribe(() => this.reloadPage());
  }
  async showConfiguration(session: SessionSummary, trigger: MatIconButton) {
    try {
      const {SessionConfiguration} = await import(
        '../session_configuration/session_configuration'
      );
      this.dialogs
        .open<
          InstanceType<typeof SessionConfiguration>,
          SessionSummary,
          'update' | 'duplicate'
        >(SessionConfiguration, {
          data: session,
          width: '900px',
          maxWidth: 'calc(100vw - 32px)',
          panelClass: 'session-configuration-dialog',
          ariaLabelledBy: 'configuration-title',
          autoFocus: '#configuration-title',
          restoreFocus: false,
        })
        .afterClosed()
        .subscribe((action) => {
          if (action)
            this.editSession(
              this.sessions.items().find((item) => item.id === session.id) ??
                session,
              action,
            );
          else trigger.focus();
        });
    } catch {
      this.dialogError('Configuration');
      trigger.focus();
    }
  }
}
