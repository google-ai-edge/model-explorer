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

import {Component, computed, inject, input, signal} from '@angular/core';
import {A11yModule} from '@angular/cdk/a11y';
import {OverlayModule} from '@angular/cdk/overlay';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {RuntimeService} from '../../../server/runtime_service';
import type {SessionSummary} from '../../../server/contracts/sessions';
import {OverlayDialog, OverlayPanel} from '../../../widgets/overlay_dialog/overlay_dialog';
@Component({
  selector: 'runner-status',
  imports: [OverlayDialog, OverlayPanel, A11yModule, OverlayModule, MatButtonModule, MatIconModule],
  templateUrl: './runner_status.ng.html',
  styleUrl: './runner_status.scss',
})
export class RunnerStatus {
  readonly runtime = inject(RuntimeService);
  readonly run = input.required<SessionSummary['runs'][number]>();
  readonly open = signal(false);
  readonly side = computed(() => (this.run().id === 'ref' ? 'Reference' : 'Target'));
  readonly runner = computed(() =>
    this.runtime.execution()?.runners.find((r) => r.role === this.run().id),
  );
  readonly uncertain = computed(() => !!this.runtime.sessions.refreshError());
  /** What the side is doing, in the server's words; a Session that never ran here is off. */
  readonly status = computed(() =>
    this.uncertain()
      ? 'Unknown'
      : this.runtime.execution()
        ? (this.runner()?.phaseText ?? 'Unknown')
        : 'Off',
  );
  readonly connection = computed(() =>
    this.uncertain()
      ? 'Unknown'
      : this.runner()?.connected === true
        ? 'Connected'
        : this.runner()?.connected === false
          ? 'Disconnected'
          : 'Unknown',
  );
}
