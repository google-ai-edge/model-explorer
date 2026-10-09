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
import type {SessionSummary} from '../../../data/contracts/session_summary';
import {RuntimeService} from '../../../data/runtime_service';
import {SessionListService} from '../../../data/session_list_service';
import {
  MODEL_SERVER_LABELS,
  sessionExecutionPhase,
} from '../../../data/session_status';

/** One Session's Model Server: its state (Failure, Starting, On, Off) and the switch that turns
 *  it on or off. The session list and the Session header use the same control. */
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
  private readonly sessions = inject(SessionListService);
  private readonly dialogs = inject(MatDialog);
  readonly state = computed(() =>
    this.runtime.modelServerState(this.session()),
  );
  readonly label = computed(() => MODEL_SERVER_LABELS[this.state()]);
  /** A request is in flight, or the Server is releasing the devices. */
  readonly busy = computed(
    () =>
      !!this.runtime.lifecyclePending()[this.session().id] ||
      sessionExecutionPhase(this.session()) === 'ending',
  );
  readonly unavailable = computed(
    () =>
      this.state() !== 'on' &&
      this.state() !== 'starting' &&
      (!!this.sessions.refreshError() ||
        !this.sessions.listing()?.generation_available),
  );
  readonly action = computed(() =>
    this.state() === 'on'
      ? 'Turn off'
      : this.state() === 'starting'
        ? 'Cancel'
        : this.state() === 'failure'
          ? 'Try again'
          : 'Turn on',
  );
  readonly error = computed(
    () =>
      this.runtime.lifecycleErrors()[this.session().id] ||
      this.session().execution?.error ||
      (this.state() === 'failure' ? this.session().notice || '' : ''),
  );
  readonly hint = computed(() =>
    sessionExecutionPhase(this.session()) === 'ending'
      ? 'Releasing both devices'
      : this.unavailable()
        ? 'Starting a Model Server is unavailable on this Server'
        : [this.error(), this.action() + ' the Model Server']
            .filter(Boolean)
            .join(' · '),
  );
  async toggle() {
    if (this.busy() || this.unavailable()) return;
    const session = this.session();
    if (this.state() === 'on') {
      const {ModelServerOff} = await import('./model_server_off');
      this.dialogs
        .open(ModelServerOff, {data: session, width: '466px'})
        .afterClosed()
        .subscribe((confirmed) => {
          if (confirmed) void this.runtime.toggleModelServer(session);
        });
    } else await this.runtime.toggleModelServer(session);
  }
}
