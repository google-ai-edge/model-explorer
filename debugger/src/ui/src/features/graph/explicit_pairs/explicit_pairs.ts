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
  output,
  signal,
} from '@angular/core';
import {ExplicitPair, ResourcePreview} from '../../../data/contracts/telemetry';
import {ReportApiService} from '../../../data/report_api_service';
import {ReportStateService} from '../../../data/report_state_service';
import {batchForwardId} from '../batch_forward';

/** Evidence-backed pairs retain their own declared observation coordinates. */
@Component({
  selector: 'explicit-pairs',
  standalone: true,
  templateUrl: './explicit_pairs.ng.html',
  styleUrl: './explicit_pairs.scss',
  host: {'[class.kv-scope]': 'scope()==="kv" && pairs().length>0'},
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ExplicitPairs {
  readonly state = inject(ReportStateService);
  private readonly api = inject(ReportApiService);
  readonly records = signal<ExplicitPair[]>([]);
  readonly error = signal('');
  readonly loading = signal(false);
  readonly scope = input<'module' | 'kv'>('module');
  readonly availability = output<boolean>();
  readonly selectedId = signal('');
  readonly previewMode = signal('original');
  readonly offset = signal(0);
  readonly previews = signal<(ResourcePreview | null)[]>([null, null]);
  readonly previewError = signal('');
  private readonly revision = signal(0);
  readonly pairs = computed(() => {
    const batch = this.state
      .session()
      ?.batches.find((b) => b.batch === this.state.batchId());
    if (!batch) return [];
    return this.records()
      .filter(
        (pair) =>
          (pair.scope ?? 'module') === this.scope() &&
          pair.observation.turn === batch.turn &&
          Object.entries(pair.runs).some(
            ([id, run]) =>
              run.runtime === batch.runtime &&
              run.observation?.phase === batch.phase &&
              run.observation.step === batch.step &&
              (batch.forward_ids
                ? batchForwardId(batch, id) != null &&
                  run.observation.forward_id === batchForwardId(batch, id)
                : batch.forward_id == null ||
                  run.observation.forward_id === batch.forward_id),
          ),
      )
      .sort((a, b) =>
        this.scope() === 'kv'
          ? (a.owner_layer ?? 0) - (b.owner_layer ?? 0) ||
            (a.kind === 'key' ? 0 : 1) - (b.kind === 'key' ? 0 : 1)
          : (a.compared_position_range?.[0] ?? 0) -
            (b.compared_position_range?.[0] ?? 0),
      );
  });
  readonly selected = computed<ExplicitPair | undefined>(
    () =>
      this.pairs().find((pair) => pair.pair_id === this.selectedId()) ??
      this.pairs()[0],
  );
  readonly shown = computed(() =>
    this.scope() === 'kv'
      ? this.selected()
        ? [this.selected()!]
        : []
      : this.pairs(),
  );
  readonly nextAvailable = computed(() =>
    this.previews().some(
      (p) => p != null && p.offset + p.values.length < p.total,
    ),
  );
  constructor() {
    effect((onCleanup) => {
      const captureId = this.state.captureId();
      this.state.revision();
      this.revision();
      const controller = new AbortController();
      onCleanup(() => controller.abort());
      this.records.set([]);
      this.error.set('');
      this.loading.set(true);
      this.api
        .explicitPairs(captureId, controller.signal)
        .then((result) => {
          if (!controller.signal.aborted) this.records.set(result);
        })
        .catch((error) => {
          if (!controller.signal.aborted) this.error.set(error.message);
        })
        .finally(() => {
          if (!controller.signal.aborted) this.loading.set(false);
        });
    });
    effect(() => this.availability.emit(this.pairs().length > 0));
    effect(() => {
      this.selected();
      this.previewMode();
      this.offset.set(0);
    });
    effect((onCleanup) => {
      const captureId = this.state.captureId(),
        pair = this.selected(),
        mode = this.previewMode(),
        offset = this.offset();
      const controller = new AbortController();
      onCleanup(() => controller.abort());
      this.previews.set([null, null]);
      this.previewError.set('');
      if (this.scope() !== 'kv' || !pair) return;
      Promise.allSettled(
        ['ref', 'target'].map((role) =>
          this.api.pairTensor(
            captureId,
            pair.pair_id,
            role,
            mode,
            offset,
            controller.signal,
          ),
        ),
      ).then((results) => {
        if (controller.signal.aborted) return;
        this.previews.set(
          results.map((result) =>
            result.status === 'fulfilled' ? result.value : null,
          ),
        );
        this.previewError.set(
          results
            .flatMap((result, index) =>
              result.status === 'rejected'
                ? [
                    `${index === 0 ? 'Reference' : 'Target'}: ${result.reason.message}`,
                  ]
                : [],
            )
            .join(' · '),
        );
      });
    });
  }
  recheck() {
    this.revision.update((value) => value + 1);
  }
  range(pair: ExplicitPair) {
    return `[${(pair.compared_position_range ?? [0, pair.processed_token_count]).join(', ')})`;
  }
  choose(event: Event) {
    this.selectedId.set((event.target as HTMLSelectElement).value);
  }
  chooseMode(event: Event) {
    this.previewMode.set((event.target as HTMLSelectElement).value);
  }
  previous() {
    this.offset.update((value) => Math.max(0, value - 32));
  }
  next() {
    this.offset.update((value) => value + 32);
  }
  view(value: ExplicitPair['runs'][string]) {
    return value.view
      .map(
        (axis) =>
          `${axis.start}:${axis.stop}${axis.step === 1 ? '' : ':' + axis.step}`,
      )
      .join(', ');
  }
  profile(value: unknown) {
    return typeof value === 'string' ? value : JSON.stringify(value);
  }
}
