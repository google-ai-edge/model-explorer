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

import {ConversationBookmarks} from '../../features/conversation/conversation_bookmarks/conversation_bookmarks';
import {OverlayModule} from '@angular/cdk/overlay';
import {A11yModule} from '@angular/cdk/a11y';
import {ConfigurationOverview} from '../../features/sessions/configuration_overview/configuration_overview';
import {ModelServerSwitch} from '../../features/sessions/model_server_switch/model_server_switch';
import {
  ChangeDetectionStrategy,
  Component,
  inject,
  viewChild,
  signal,
  computed,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatMenuModule, MatMenuTrigger} from '@angular/material/menu';
import {MatTooltipModule} from '@angular/material/tooltip';
import {MatDialog} from '@angular/material/dialog';
import {MatSnackBar} from '@angular/material/snack-bar';
import {ReportStateService} from '../../data/report_state_service';
import {WorkspaceStateService} from '../workspace_state_service';
import {ThemeService} from '../../theme/theme_service';
import {OverlayDialog, OverlayPanel} from '../../shared/overlay_dialog/overlay_dialog';
@Component({
  selector: 'app-toolbar',
  imports: [
    OverlayDialog,
    OverlayPanel,
    ConversationBookmarks,
    OverlayModule,
    A11yModule,
    ConfigurationOverview,
    ModelServerSwitch,
    MatButtonModule,
    MatIconModule,
    MatMenuModule,
    MatTooltipModule,
  ],
  templateUrl: './app_toolbar.ng.html',
  styleUrl: './app_toolbar.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AppToolbar {
  readonly state = inject(ReportStateService);
  readonly workspace = inject(WorkspaceStateService);
  readonly theme = inject(ThemeService);
  private readonly dialog = inject(MatDialog);
  private readonly snackBar = inject(MatSnackBar);
  private dialogFailure(label: string) {
    this.snackBar
      .open(label + ' could not be loaded. Reload the page and try again.', 'Reload')
      .onAction()
      .subscribe(() => window.location.reload());
  }
  private readonly aboutTrigger = viewChild<MatMenuTrigger>('aboutTrigger');
  readonly themes = ['system', 'light', 'dark'] as const;
  readonly configurationOpen = signal(false);
  readonly chatOpen = signal(false);
  readonly chatQuery = signal('');
  readonly filteredChats = computed(() =>
    this.workspace
      .chats()
      .filter((c) =>
        c.name.toLocaleLowerCase().includes(this.chatQuery().trim().toLocaleLowerCase()),
      ),
  );
  openChats() {
    this.chatQuery.set('');
    this.chatOpen.set(!this.chatOpen());
  }
  chatTriggerKey(event: KeyboardEvent) {
    if (['ArrowDown', 'ArrowUp'].includes(event.key)) {
      event.preventDefault();
      this.chatQuery.set('');
      this.chatOpen.set(true);
    }
  }
  chatKey(event: KeyboardEvent) {
    if (event.isComposing) {
      return;
    }
    const panel = (event.target as HTMLElement).closest('.chat-picker');
    const options = Array.from(panel?.querySelectorAll<HTMLButtonElement>('[role=option]') ?? []);
    if (
      ['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key) &&
      (!(event.target instanceof HTMLInputElement) || event.key.startsWith('Arrow'))
    ) {
      event.preventDefault();
      const index = options.indexOf(event.target as HTMLButtonElement);
      const next =
        event.key === 'Home'
          ? 0
          : event.key === 'End'
            ? options.length - 1
            : index < 0
              ? event.key === 'ArrowUp'
                ? options.length - 1
                : 0
              : (index + (event.key === 'ArrowUp' ? -1 : 1) + options.length) % options.length;
      options[next]?.focus();
    } else if (
      event.key === 'Enter' &&
      event.target instanceof HTMLInputElement &&
      this.filteredChats().length
    ) {
      event.preventDefault();
      this.selectChat(this.filteredChats()[0].id);
    }
  }
  selectChat(id: string) {
    this.chatOpen.set(false);
    void this.workspace.chooseChat(id);
  }

  async newChat() {
    this.chatOpen.set(false);
    try {
      const {ChatConfiguration} = await import(
        '../../features/conversation/chat_configuration/chat_configuration'
      );
      this.dialog.open(ChatConfiguration, {
        width: '700px',
        maxWidth: 'calc(100vw - 24px)',
        data: {create: true},
        ariaLabelledBy: 'chat-configuration-title',
        autoFocus: '#chat-configuration-title',
      });
    } catch {
      this.dialogFailure('Chat configuration');
    }
  }
  async settings() {
    try {
      const {AppSettings} = await import('../app_settings/app_settings');
      this.dialog.open(AppSettings, {width: '466px', maxWidth: 'calc(100vw - 32px)'});
    } catch {
      this.dialogFailure('Settings');
    }
  }
  async about() {
    try {
      const {AboutDialog} = await import('../about_dialog/about_dialog');
      this.dialog
        .open(AboutDialog, {width: '560px', maxWidth: 'calc(100vw - 32px)', restoreFocus: false})
        .afterClosed()
        .subscribe(() => this.aboutTrigger()?.focus('keyboard'));
    } catch {
      this.dialogFailure('About');
      this.aboutTrigger()?.focus('keyboard');
    }
  }
}
