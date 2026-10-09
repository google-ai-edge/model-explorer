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

import {Component, inject, output, TemplateRef, viewChild} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatDialog, MatDialogModule} from '@angular/material/dialog';
import {MatSnackBar} from '@angular/material/snack-bar';
import {MatTabsModule} from '@angular/material/tabs';
import {WorkspaceStateService} from '../../../app/workspace_state_service';
import {GENERATION_GROUPS, GENERATION_LABELS} from '../chat_configuration';
import {SessionConfiguration} from '../session_configuration/session_configuration';
@Component({
  selector: 'configuration-overview',
  imports: [
    MatTabsModule,
    MatButtonModule,
    MatDialogModule,
    SessionConfiguration,
  ],
  templateUrl: './configuration_overview.ng.html',
  styleUrl: './configuration_overview.scss',
})
export class ConfigurationOverview {
  private readonly snackBar = inject(MatSnackBar);
  readonly workspace = inject(WorkspaceStateService);
  readonly dialog = inject(MatDialog);
  readonly closed = output<void>();
  get parent() {
    return this.workspace.context().parentSession;
  }
  get chat() {
    return this.workspace.context().activeRecord;
  }
  readonly groups = GENERATION_GROUPS.filter((g) => g.name !== 'System prompt');
  readonly labels = GENERATION_LABELS;
  readonly prompt = viewChild.required<TemplateRef<unknown>>('prompt');
  value(side: string, key: string) {
    const config = this.chat?.generation;
    const raw = config?.[side as 'ref' | 'target']?.[key];
    if (!config) return 'Not recorded';
    return raw == null || raw === ''
      ? 'Default'
      : key === 'thinking'
        ? ({on: 'Enabled', off: 'Disabled'}[String(raw)] ?? 'Default')
        : String(raw);
  }
  showPrompt() {
    this.dialog.open(this.prompt(), {
      width: '560px',
      maxWidth: 'calc(100vw - 48px)',
    });
  }
  async edit() {
    try {
      const {ChatConfiguration} = await import(
        '../../conversation/chat_configuration/chat_configuration'
      );
      this.closed.emit();
      this.dialog.open(ChatConfiguration, {
        width: '700px',
        maxWidth: 'calc(100vw - 24px)',
        ariaLabelledBy: 'chat-configuration-title',
        autoFocus: '#chat-configuration-title',
      });
    } catch {
      this.snackBar
        .open(
          'Chat configuration could not be loaded. Reload the page and try again.',
          'Reload',
        )
        .onAction()
        .subscribe(() => window.location.reload());
    }
  }
}
