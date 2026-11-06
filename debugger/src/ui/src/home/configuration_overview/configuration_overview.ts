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

import {Component, inject, TemplateRef, viewChild} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatDialog, MatDialogModule} from '@angular/material/dialog';
import {MatTabsModule} from '@angular/material/tabs';
import {WorkspaceStateService} from '../../state/workspace_state_service';
import {chatOptionGroups, switchLabel} from '../chat_configuration';
import {SessionConfiguration} from '../session_configuration/session_configuration';

@Component({
  selector: 'configuration-overview',
  imports: [MatTabsModule, MatButtonModule, MatDialogModule, SessionConfiguration],
  templateUrl: './configuration_overview.ng.html',
  styleUrl: './configuration_overview.scss',
})
export class ConfigurationOverview {
  readonly workspace = inject(WorkspaceStateService);
  readonly dialog = inject(MatDialog);
  readonly prompt = viewChild.required<TemplateRef<unknown>>('prompt');
  readonly sides = ['ref', 'target'] as const;

  /** The Session. */
  get session() {
    return this.workspace.session();
  }
  /** The chat the page shows, whose settings the Chat tab shows; null on the New chat form. */
  get chat() {
    return this.workspace.chat();
  }
  /** The Chat settings as the server describes them (option_catalog.py). */
  private get options() {
    return this.workspace.runtime.capability()?.chat_options ?? [];
  }
  get groups() {
    return chatOptionGroups(this.options.filter((option) => option.key !== 'systemPrompt'));
  }
  label(key: string): string {
    return this.options.find((option) => option.key === key)?.label ?? key;
  }
  /**
   * Chat settings that differ between the sides. Differences are shown, never blocked: they
   * make a numeric difference attributable to the configuration, not only to the model.
   */
  differs(key: string): boolean {
    return !!this.chat && this.value('ref', key) !== this.value('target', key);
  }
  get differingKeys(): string[] {
    return this.options.map((option) => option.key).filter((key) => this.differs(key));
  }
  get differenceNote(): string {
    const keys = this.differingKeys;
    if (!keys.length) {
      return '';
    }
    return (
      (keys.length === 1 ? 'One Chat setting differs' : keys.length + ' Chat settings differ') +
      ' between Reference and Target: ' +
      keys.map((key) => this.label(key)).join(', ') +
      '. Output differences may come from these settings, not only from the model.'
    );
  }
  value(side: 'ref' | 'target', key: string): string {
    const config = this.chat?.generation;
    if (!config) {
      return 'Not recorded';
    }
    const raw = config[side]?.[key];
    const option = this.options.find((option) => option.key === key);
    return raw == null || raw === ''
      ? 'Default'
      : option?.type === 'switch'
        ? switchLabel(raw)
        : (option?.choice_labels?.[String(raw)] ?? String(raw));
  }
  showPrompt(): void {
    this.dialog.open(this.prompt(), {width: '560px', maxWidth: 'calc(100vw - 48px)'});
  }
}
