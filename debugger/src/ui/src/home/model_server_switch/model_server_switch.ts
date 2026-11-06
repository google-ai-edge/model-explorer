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

import {Component, computed, inject, input} from '@angular/core';
import {MatDialog} from '@angular/material/dialog';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import type {SessionSummary} from '../../server/contracts/sessions';
import {RuntimeService} from '../../server/runtime_service';

/**
 * One Session's Model Server: its state (Starting, On, or how its Session ended) and the
 * switch that turns it off or cancels its start; a Session starts as it is created, so nothing
 * turns one on. What it shows and whether it can be clicked are the server's `model_server`.
 * The session list and the Session header use the same control.
 */
@Component({
  selector: 'model-server-switch',
  imports: [MatIconModule, MatTooltipModule],
  templateUrl: './model_server_switch.ng.html',
  styleUrl: './model_server_switch.scss',
})
export class ModelServerSwitch {
  readonly session = input.required<SessionSummary>();
  /** Spell out "Model Server" where no column header names the control. */
  readonly named = input(false);
  private readonly runtime = inject(RuntimeService);
  private readonly dialogs = inject(MatDialog);
  readonly view = computed(() => this.session().model_server);
  readonly state = computed(() => this.view().state);
  readonly label = computed(() => this.view().text);
  /** This page's request is in flight. */
  readonly busy = computed(() => !!this.runtime.lifecyclePending()[this.session().id]);
  /** Whether the switch can be clicked: the server allows it. */
  readonly enabled = computed(() => !this.busy() && !!this.view().button?.allowed);
  /**
   * A refusal of this page's last request, or why the Session ended with no turn (its start
   * failed, say).
   */
  readonly error = computed(
    () => this.runtime.lifecycleErrors()[this.session().id] || this.view().error,
  );
  readonly shownError = computed(() =>
    this.runtime.lifecycleErrors()[this.session().id] || this.state() === 'ended'
      ? this.error()
      : '',
  );
  readonly hint = computed(() => {
    const button = this.view().button;
    if (!button) {
      return this.error() || this.label();
    }
    return !button.allowed
      ? button.reason
      : [this.error(), button.label + ' the Model Server'].filter(Boolean).join(' · ');
  });
  async toggle(): Promise<void> {
    if (!this.enabled()) {
      return;
    }
    const session = this.session();
    if (this.state() === 'on') {
      const {ModelServerOff} = await import('./model_server_off');
      this.dialogs
        .open(ModelServerOff, {data: session, width: '466px'})
        .afterClosed()
        .subscribe((confirmed) => {
          if (confirmed) {
            void this.runtime.closeModelServer(session);
          }
        });
    } else {
      await this.runtime.closeModelServer(session);
    }
  }
}
