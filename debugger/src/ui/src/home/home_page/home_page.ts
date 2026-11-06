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
  DestroyRef,
  ElementRef,
  inject,
  signal,
} from '@angular/core';
import {takeUntilDestroyed} from '@angular/core/rxjs-interop';
import {MatButtonModule, MatIconButton} from '@angular/material/button';
import {MatDialog} from '@angular/material/dialog';
import {MatIconModule} from '@angular/material/icon';
import {MatMenuModule} from '@angular/material/menu';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatSnackBar} from '@angular/material/snack-bar';
import {MatTooltipModule} from '@angular/material/tooltip';
import {SessionSummary} from '../../server/contracts/sessions';
import {SessionListService, type SessionImport} from '../../server/session_list_service';
import {PreferencesService} from '../../state/preferences_service';
import {WorkspaceStateService} from '../../state/workspace_state_service';
import {ModelServerSwitch} from '../model_server_switch/model_server_switch';
import {SessionEditor} from '../session_editor/session_editor';
import {WelcomeCard} from '../welcome_card/welcome_card';

/** How long a row says Exporting… on this page's word alone: the list's polls say it after. */
const EXPORT_HANDOVER_MS = 4000;
/** The server's words for a Session whose export is being sent (session_archive.EXPORTING). */
const EXPORTING = 'This Session is being exported; wait until its file is saved.';

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
    MatProgressBarModule,
    MatProgressSpinnerModule,
  ],
  templateUrl: './home_page.ng.html',
  styleUrl: './home_page.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class HomePage {
  readonly preferences = inject(PreferencesService);
  readonly sessions = inject(SessionListService);
  readonly workspace = inject(WorkspaceStateService);
  private readonly clipboard = inject(Clipboard);
  private readonly dialogs = inject(MatDialog);
  private readonly snackBar = inject(MatSnackBar);
  private readonly element: ElementRef<HTMLElement> = inject(ElementRef);
  private readonly destroyRef = inject(DestroyRef);
  /**
   * Whether the Session editor is open; it only creates: a Session's configuration never
   * changes once created, only its name.
   */
  readonly editor = signal(false);
  readonly editing = signal<string | null>(null);
  readonly saving = signal(false);
  readonly renameError = signal('');
  /** Exports this page started, until the list says so itself (`exporting`). */
  private readonly exportsStarted = signal<ReadonlySet<string>>(new Set());
  private readonly timers = new Set<ReturnType<typeof setTimeout>>();
  private renameTrigger: MatIconButton | null = null;

  constructor() {
    this.destroyRef.onDestroy(() => this.timers.forEach((timer) => clearTimeout(timer)));
  }

  reloadPage(): void {
    window.location.reload();
  }

  newSession(): void {
    this.editor.set(true);
  }

  /** A created Session stays in the list while its Model Server starts; Open unlocks once it is on. */
  sessionCreated(): void {
    this.snackBar.open('Session created. Its Model Server is starting.', 'Dismiss', {
      duration: 5000,
    });
  }

  finishEditor(): void {
    this.editor.set(false);
    // The list is recreated once the editor closes.
    requestAnimationFrame(() =>
      this.element.nativeElement.querySelector<HTMLButtonElement>('.new-session')?.focus(),
    );
  }

  openSession(session: SessionSummary): void {
    if (session.actions.enter.allowed && !this.exporting(session)) {
      this.workspace.open(session.id);
    }
  }

  /**
   * Whether a Session's export is being sent: the server says so, or this page just asked. While
   * it is, the row says Exporting…, and Export and Open wait.
   */
  exporting(session: SessionSummary): boolean {
    return !!session.exporting || this.exportsStarted().has(session.id);
  }

  /**
   * Saves a Session that has ended as one file, `<name>.mdsession`, where the browser saves
   * downloads; the server streams it, and says it is exporting until the stream ends.
   */
  exportSession(session: SessionSummary): void {
    if (!session.actions.export.allowed || this.exporting(session)) {
      return;
    }
    const link = this.element.nativeElement.ownerDocument.createElement('a');
    link.href = this.sessions.exportUrl(session.id);
    link.download = '';
    link.click();
    const id = session.id;
    this.exportsStarted.update((ids) => new Set(ids).add(id));
    // The list says so itself from the next poll on, asked for soon.
    this.later(() => void this.sessions.load(true), 500);
    this.later(
      () =>
        this.exportsStarted.update((ids) => {
          const left = new Set(ids);
          left.delete(id);
          return left;
        }),
      EXPORT_HANDOVER_MS,
    );
  }

  /** Runs `callback` after `ms`, unless the page is gone by then. */
  private later(callback: () => void, ms: number): void {
    const timer = setTimeout(() => {
      this.timers.delete(timer);
      callback();
    }, ms);
    this.timers.add(timer);
  }

  /** Opens the browser's file picker for a Session file another computer exported. */
  chooseImport(input: HTMLInputElement): void {
    input.value = '';
    input.click();
  }

  /** Imports the file chosen; its row shows the progress, then the Session joins the list. */
  async importFile(input: HTMLInputElement): Promise<void> {
    const file = input.files?.[0];
    input.value = '';
    if (!file) {
      return;
    }
    try {
      const imported = await this.sessions.importSession(file);
      this.snackBar.open(`Imported ${imported.name}. Open it to debug.`, 'Dismiss', {
        duration: 5000,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.snackBar.open('Import failed: ' + message, 'Dismiss');
    }
  }

  importPercent(importing: SessionImport): number {
    return importing.total ? Math.floor((importing.loaded * 100) / importing.total) : 0;
  }

  /** What the import row says: the upload's share, then the server unpacking it. */
  importText(importing: SessionImport): string {
    return importing.loaded < importing.total
      ? `Uploading · ${this.importPercent(importing)}%`
      : 'Unpacking…';
  }

  async deleteSession(session: SessionSummary, trigger: MatIconButton): Promise<void> {
    try {
      const {SessionDelete} = await import('../session_delete/session_delete');
      this.dialogs
        .open(SessionDelete, {data: session, width: '466px'})
        .afterClosed()
        .pipe(takeUntilDestroyed(this.destroyRef))
        .subscribe((deleted) => {
          if (!deleted) {
            trigger.focus();
          } else {
            requestAnimationFrame(() =>
              this.element.nativeElement.querySelector<HTMLButtonElement>('.new-session')?.focus(),
            );
          }
        });
    } catch {
      this.dialogError('Delete dialog');
      trigger.focus();
    }
  }

  /** The one way into a Session; the server says whether there is something to open. */
  openHint(session: SessionSummary): string {
    return !session.actions.enter.allowed
      ? session.actions.enter.reason
      : this.exporting(session)
        ? EXPORTING
        : 'Open session';
  }

  /** Why Export is greyed out now, or '' when it can be clicked. */
  exportHint(session: SessionSummary): string {
    return !session.actions.export.allowed
      ? session.actions.export.reason
      : this.exporting(session)
        ? EXPORTING
        : '';
  }

  /** The server's words for a tensor capture being prepared. */
  preparation(session: SessionSummary): string {
    const preparation = session.tap_preparation;
    return preparation?.state === 'preparing' ? preparation.text : '';
  }

  /** Each side's phase and latest progress line while its Model Server starts or runs. */
  sides(session: SessionSummary): Array<{role: string; side: string; text: string}> {
    const state = session.model_server.state;
    if (state !== 'starting' && state !== 'on') {
      return [];
    }
    return (session.execution?.runners ?? []).map((runner) => ({
      role: runner.role,
      side: runner.role === 'ref' ? 'Reference' : 'Target',
      // A progress line that only repeats the phase, or begins with it, says it once.
      text: !runner.progress
        ? runner.phaseText
        : runner.progress.startsWith(runner.phaseText)
          ? runner.progress
          : runner.phaseText + ' · ' + runner.progress,
    }));
  }

  startRename(session: SessionSummary, trigger: MatIconButton): void {
    if (this.saving()) {
      return;
    }
    this.renameTrigger = trigger;
    this.editing.set(session.id);
    this.renameError.set('');
    requestAnimationFrame(() => {
      const input =
        this.element.nativeElement.querySelector<HTMLInputElement>('.session-name-input');
      input?.focus();
      input?.select();
    });
  }

  async finishRename(
    session: SessionSummary,
    input: HTMLInputElement,
    save: boolean,
    restoreFocus = false,
  ): Promise<void> {
    if (this.editing() !== session.id || this.saving()) {
      return;
    }
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
      } else {
        this.editing.set(null);
      }
      return;
    }
    this.saving.set(true);
    this.renameError.set('');
    try {
      if (name !== session.name) {
        await this.sessions.rename(session.id, name);
      }
      if (this.editing() !== session.id) {
        return;
      }
      const document = this.element.nativeElement.ownerDocument;
      // Keep a completed blur save from dropping focus when its input disappears.
      const returnFocus =
        input.isConnected &&
        (restoreFocus ||
          document.activeElement === input ||
          document.activeElement === document.body);
      this.editing.set(null);
      if (returnFocus) {
        requestAnimationFrame(() => trigger?.focus());
      }
    } catch (error) {
      if (this.editing() === session.id) {
        this.renameError.set(error instanceof Error ? error.message : String(error));
        if (input.isConnected) {
          input.focus();
        }
      }
    } finally {
      this.saving.set(false);
    }
  }

  renameKey(event: KeyboardEvent, session: SessionSummary, input: HTMLInputElement): void {
    if (event.isComposing) {
      return;
    }
    if (event.key === 'Enter' || event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      void this.finishRename(session, input, event.key === 'Enter', true);
    }
  }

  copy(session: SessionSummary): void {
    const copied = this.clipboard.copy(session.id);
    this.snackBar.open(copied ? 'Session ID copied' : 'Could not copy session ID', 'Dismiss', {
      duration: 3000,
    });
  }

  private dialogError(label: string): void {
    this.snackBar
      .open(label + ' could not be loaded. Reload the page and try again.', 'Reload')
      .onAction()
      .subscribe(() => this.reloadPage());
  }

  async showConfiguration(session: SessionSummary, trigger: MatIconButton): Promise<void> {
    try {
      const {SessionConfiguration} = await import('../session_configuration/session_configuration');
      this.dialogs
        .open<InstanceType<typeof SessionConfiguration>, SessionSummary>(SessionConfiguration, {
          data: session,
          width: '900px',
          maxWidth: 'calc(100vw - 32px)',
          panelClass: 'session-configuration-dialog',
          ariaLabelledBy: 'configuration-title',
          autoFocus: '#configuration-title',
          restoreFocus: false,
        })
        .afterClosed()
        .pipe(takeUntilDestroyed(this.destroyRef))
        .subscribe(() => trigger.focus());
    } catch {
      this.dialogError('Configuration');
      trigger.focus();
    }
  }
}
