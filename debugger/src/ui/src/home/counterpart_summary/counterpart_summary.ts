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

import {Component, effect, inject, input, OnDestroy, signal} from '@angular/core';
import {ArtifactsApi} from '../../server/artifacts_api';
import type {TapCounterparts} from '../../server/contracts/artifacts';

/**
 * What a PyTorch side captures beside its anchors for the outputs chosen of the other side's
 * LiteRT-LM model, as the server pairs them (GET /api/artifacts/{id}/counterparts): each module
 * it captures for one, and why an output chosen has none.
 */
@Component({
  selector: 'counterpart-summary',
  templateUrl: './counterpart_summary.ng.html',
  styleUrl: './counterpart_summary.scss',
})
export class CounterpartSummary implements OnDestroy {
  /** The LiteRT-LM model whose outputs are chosen. */
  readonly artifact = input.required<string>();
  /** The PyTorch model of this side. */
  readonly pytorch = input.required<string>();
  /** The outputs chosen of `artifact`. */
  readonly points = input<string[]>([]);
  readonly pairing = signal<TapCounterparts | null>(null);
  readonly error = signal('');
  private readonly api = inject(ArtifactsApi);
  private destroyed = false;
  private request = 0;
  constructor() {
    effect(() => void this.load(this.artifact(), this.pytorch(), this.points()));
  }
  ngOnDestroy() {
    this.destroyed = true;
  }
  /** The modules it captures for the outputs chosen, and the outputs paired with none. */
  modules() {
    return (
      this.pairing()?.points.filter((p) => p.paired === 'module' || p.paired === 'manual') ?? []
    );
  }
  unpaired() {
    return this.pairing()?.points.filter((p) => !p.paired && p.text) ?? [];
  }
  private async load(artifact: string, pytorch: string, points: string[]) {
    const request = ++this.request;
    this.error.set('');
    if (!artifact || !pytorch || !points.length) {
      this.pairing.set(null);
      return;
    }
    try {
      const found = await this.api.tapCounterparts(artifact, pytorch, points);
      if (request === this.request && !this.destroyed) {
        this.pairing.set(found);
      }
    } catch (error) {
      if (request === this.request && !this.destroyed) {
        this.pairing.set(null);
        this.error.set(error instanceof Error ? error.message : String(error));
      }
    }
  }
}
