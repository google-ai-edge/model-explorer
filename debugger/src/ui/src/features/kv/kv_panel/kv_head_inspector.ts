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
  effect,
  inject,
  input,
  model,
  signal,
  untracked,
} from '@angular/core';
import type {KvHeadEvidence} from '../../../data/contracts/kv';
import {ReportApiService} from '../../../data/report_api_service';
import {ReportStateService} from '../../../data/report_state_service';
import {KvHeadDetails} from './kv_head_details';

export interface KvHeadSelection {
  turn: number;
  contextId: string;
  layer: number;
  kind: string;
  position: number;
  head: number;
}

/** One head owns its request; the enclosing inspector owns its persistent UI state. */
@Component({
  selector: 'kv-head-inspector',
  standalone: true,
  imports: [KvHeadDetails],
  template: `<kv-head-details
    [evidence]="evidence()"
    [loading]="loading()"
    [error]="error()"
    (retry)="retry.update(increment)"
    [(view)]="view"
    [(numericalExpanded)]="numericalExpanded"
    [(channelsExpanded)]="channelsExpanded"
  />`,
  styles: ':host{display:block;min-width:0;}',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class KvHeadInspector {
  private readonly api = inject(ReportApiService);
  private readonly state = inject(ReportStateService);
  readonly selection = input<KvHeadSelection | null>(null);
  readonly active = input(false);
  readonly view = model<'index' | 'scatter'>('index');
  readonly numericalExpanded = model(true);
  readonly channelsExpanded = model(true);
  readonly evidence = signal<KvHeadEvidence | null>(null);
  readonly loading = signal(false);
  readonly error = signal('');
  readonly retry = signal(0);
  readonly increment = (value: number) => value + 1;
  private readonly requestKey = computed(() => {
    const selection = this.selection();
    return JSON.stringify([
      this.state.captureId(),
      selection && [
        selection.turn,
        selection.contextId,
        selection.layer,
        selection.kind,
        selection.position,
        selection.head,
      ],
    ]);
  });

  constructor() {
    effect((onCleanup) => {
      this.requestKey();
      this.retry();
      const captureId = this.state.captureId(),
        active = this.active(),
        selection = untracked(this.selection);
      const controller = new AbortController();
      onCleanup(() => controller.abort());
      this.evidence.set(null);
      this.error.set('');
      this.loading.set(false);
      if (!active || !selection) return;
      this.loading.set(true);
      untracked(() =>
        this.api.kvHead(
          captureId,
          selection.turn,
          selection.contextId,
          selection.layer,
          selection.kind,
          selection.position,
          selection.head,
          null,
          controller.signal,
        ),
      )
        .then((value) => {
          if (controller.signal.aborted) return;
          const returned = value.selection;
          if (
            returned.turn !== selection.turn ||
            returned.context_id !== selection.contextId ||
            returned.layer !== selection.layer ||
            returned.kind !== selection.kind ||
            returned.position !== selection.position ||
            returned.head !== selection.head
          ) {
            this.error.set(
              'Head values do not match this selection. Please retry.',
            );
            return;
          }
          this.evidence.set(value);
        })
        .catch(() => {
          if (!controller.signal.aborted)
            this.error.set('Unable to load this head. Please retry.');
        })
        .finally(() => {
          if (!controller.signal.aborted) this.loading.set(false);
        });
    });
  }
}
