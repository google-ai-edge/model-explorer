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

import {A11yModule} from '@angular/cdk/a11y';
import {OverlayModule} from '@angular/cdk/overlay';
import {Component, computed, inject, input, signal} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {SessionSummary} from '../../../data/contracts/session_summary';
import {RuntimeService} from '../../../data/runtime_service';
import {
  OverlayDialog,
  OverlayPanel,
} from '../../../shared/overlay_dialog/overlay_dialog';
@Component({
  selector: 'runner-status',
  imports: [
    OverlayDialog,
    OverlayPanel,
    A11yModule,
    OverlayModule,
    MatButtonModule,
    MatIconModule,
  ],
  templateUrl: './runner_status.ng.html',
  styleUrl: './runner_status.scss',
})
export class RunnerStatus {
  readonly runtime = inject(RuntimeService);
  readonly run = input.required<SessionSummary['runs'][number]>();
  readonly open = signal(false);
  readonly side = computed(() =>
    this.run().id === 'ref' ? 'Reference' : 'Target',
  );
  readonly runner = computed(() =>
    this.runtime.execution()?.runners.find((r) => r.role === this.run().id),
  );
  readonly uncertain = computed(() => !!this.runtime.sessions.refreshError());
  readonly status = computed(() => {
    if (this.uncertain()) return 'Unknown';
    const execution = this.runtime.execution();
    // Idle sides use the Model Server words of the switch; a live side shows what it is doing.
    if (
      !execution ||
      execution.phase === 'inactive' ||
      execution.phase === 'ended'
    )
      return 'Off';
    if (execution.phase === 'interrupted') return 'Failure';
    if (execution.phase === 'ending') return 'Ending';
    const phase = this.runner()?.phase;
    return (
      (
        {
          starting: 'Starting',
          connecting: 'Connecting',
          loading: 'Loading model',
          initializing: 'Initializing',
          ready: 'Ready',
          active: 'Ready',
          checking_input: 'Checking input',
          generating: 'Generating',
          running: 'Generating',
          transferring: 'Transferring capture',
          stopping: 'Stopping',
          restoring: 'Restoring conversation',
          ending: 'Ending',
          releasing: 'Releasing',
          ended: 'Off',
          interrupted: 'Failure',
          disconnected: 'Disconnected',
          failed: 'Failure',
          unknown: 'Unknown',
        } as Record<string, string>
      )[phase ?? ''] ??
      (execution.phase === 'starting' ? 'Starting' : 'Unknown')
    );
  });
  readonly connection = computed(() =>
    this.uncertain()
      ? 'Unknown'
      : this.runner()?.connected === true
        ? 'Connected'
        : this.runner()?.connected === false
          ? 'Disconnected'
          : 'Unknown',
  );
  timestamp(value: string | undefined) {
    return value ? new Date(value).toLocaleTimeString() : 'Unknown';
  }
}
