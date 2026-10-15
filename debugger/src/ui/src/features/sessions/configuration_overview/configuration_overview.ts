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

import {Component, inject, output, viewChild, TemplateRef} from '@angular/core';
import {MatTabsModule} from '@angular/material/tabs';
import {MatButtonModule} from '@angular/material/button';
import {MatDialog, MatDialogModule} from '@angular/material/dialog';
import {SessionConfiguration} from '../session_configuration/session_configuration';
import {MatSnackBar} from '@angular/material/snack-bar';
import {WorkspaceStateService} from '../../../app/workspace_state_service';
import {
  GENERATION_GROUPS,
  GENERATION_LABELS,
  GENERATION_SWITCHES,
  switchLabel,
} from '../chat_configuration';
@Component({
  selector: 'configuration-overview',
  imports: [MatTabsModule, MatButtonModule, MatDialogModule, SessionConfiguration],
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
  readonly groups = GENERATION_GROUPS.map((g) => ({
    ...g,
    keys: g.keys.filter((key) => key !== 'systemPrompt'),
  }));
  readonly labels = GENERATION_LABELS;
  readonly prompt = viewChild.required<TemplateRef<unknown>>('prompt');
  /** Chat settings that differ between the sides. Differences are shown, never blocked: they
   * make a numeric difference attributable to the configuration, not only to the model. */
  differs(key: string) {
    return !!this.chat?.generation && this.value('ref', key) !== this.value('target', key);
  }
  get differingKeys() {
    return GENERATION_GROUPS.flatMap((g) => g.keys).filter((key) => this.differs(key));
  }
  get differenceNote() {
    const keys = this.differingKeys;
    if (!keys.length) return '';
    return (
      (keys.length === 1 ? 'One Chat setting differs' : keys.length + ' Chat settings differ') +
      ' between Reference and Target: ' +
      keys.map((key) => GENERATION_LABELS[key]).join(', ') +
      '. Output differences may come from these settings, not only from the model.'
    );
  }
  value(side: string, key: string) {
    const config = this.chat?.generation;
    const raw = config?.[side as 'ref' | 'target']?.[key];
    if (!config) return 'Not recorded';
    return raw == null || raw === ''
      ? 'Default'
      : GENERATION_SWITCHES.includes(key)
        ? switchLabel(raw)
        : String(raw);
  }
  showPrompt() {
    this.dialog.open(this.prompt(), {width: '560px', maxWidth: 'calc(100vw - 48px)'});
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
        .open('Chat configuration could not be loaded. Reload the page and try again.', 'Reload')
        .onAction()
        .subscribe(() => window.location.reload());
    }
  }
}
