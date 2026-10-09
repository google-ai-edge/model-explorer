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
  ChangeDetectionStrategy,
  Component,
  computed,
  DestroyRef,
  effect,
  inject,
  untracked,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {ReportStateService} from '../../data/report_state_service';
import {ConversationPanel} from '../../features/conversation/conversation_panel/conversation_panel';
import {GraphWorkspace} from '../../features/graph/graph_workspace/graph_workspace';
import {KvPanel} from '../../features/kv/kv_panel/kv_panel';
import {HomePage} from '../../features/sessions/home_page/home_page';
import {AppToolbar} from '../app_toolbar/app_toolbar';
import {WorkspaceStateService} from '../workspace_state_service';
@Component({
  selector: 'report-app',
  standalone: true,
  imports: [
    MatButtonModule,
    HomePage,
    AppToolbar,
    GraphWorkspace,
    KvPanel,
    ConversationPanel,
  ],
  templateUrl: './report.ng.html',
  styleUrl: './report.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ReportPage {
  constructor() {
    effect(() => {
      const active =
        !this.workspace.home() &&
        this.workspace.mode() === 'Debug' &&
        this.workspace.view() === 'Graph Diff';
      untracked(() => this.state.setGraphActive(active));
    });
    inject(DestroyRef).onDestroy(() => this.state.setGraphActive(false));
  }
  reloadPage() {
    window.location.reload();
  }
  readonly state = inject(ReportStateService);
  readonly workspace = inject(WorkspaceStateService);
  readonly conversationKey = computed(() => {
    const {parentSessionId, activeChatId} = this.workspace.context();
    return JSON.stringify([parentSessionId, activeChatId]);
  });
}
