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

import {Component, ElementRef, inject, signal} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {
  MAT_DIALOG_DATA,
  MatDialogModule,
  MatDialogRef,
} from '@angular/material/dialog';
import {SessionSummary} from '../../../data/contracts/session_summary';
import {SessionListService} from '../../../data/session_list_service';
import {sessionConfigurationLocked} from '../../../data/session_status';
@Component({
  selector: 'session-delete',
  imports: [MatDialogModule, MatButtonModule],
  templateUrl: './session_delete.ng.html',
  styleUrl: './session_delete.scss',
})
export class SessionDelete {
  private readonly element = inject<ElementRef<HTMLElement>>(ElementRef);
  readonly session = inject<SessionSummary>(MAT_DIALOG_DATA);
  readonly ref = inject(MatDialogRef<SessionDelete>);
  readonly sessions = inject(SessionListService);
  readonly saving = signal(false);
  readonly error = signal('');
  async remove() {
    if (this.saving()) return;
    const current = this.sessions
      .items()
      .find((item) => item.id === this.session.id);
    if (
      !current ||
      !this.sessions.listing()?.capabilities.delete ||
      sessionConfigurationLocked(current)
    ) {
      this.error.set(
        'This session cannot be deleted in its current state. Cancel and check the session list.',
      );
      return;
    }
    this.saving.set(true);
    this.ref.disableClose = true;
    try {
      await this.sessions.manage('delete', {id: this.session.id});
      this.ref.close(true);
    } catch (e) {
      this.error.set((e as Error).message);
      requestAnimationFrame(() =>
        this.element.nativeElement
          .querySelector<HTMLElement>('[role="alert"]')
          ?.focus(),
      );
    } finally {
      this.saving.set(false);
      this.ref.disableClose = false;
    }
  }
}
