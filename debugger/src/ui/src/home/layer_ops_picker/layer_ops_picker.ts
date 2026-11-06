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

import {Component, effect, inject, input, output, signal, OnDestroy} from '@angular/core';
import {ArtifactsApi} from '../../server/artifacts_api';
import type {LayerOps} from '../../server/contracts/artifacts';

/**
 * What else to see inside a layer: one list of the operators of the two models' decoder layer,
 * by block, as the server lists them (GET /api/models/ops). Ticking one captures it in every layer
 * that has it, on both sides, beside the anchors; the server says where a side lacks one and
 * why, what each adds per generated token, the total, and why those ticked cannot be captured.
 */
@Component({
  selector: 'layer-ops-picker',
  templateUrl: './layer_ops_picker.ng.html',
  styleUrl: './layer_ops_picker.scss',
})
export class LayerOpsPicker implements OnDestroy {
  /** The Reference side's model. */
  readonly ref = input.required<string>();
  /** The Target side's model. */
  readonly target = input.required<string>();
  /** The operators ticked. */
  readonly selected = input<string[]>([]);
  readonly disabled = input(false);
  readonly selectionChange = output<string[]>();
  /** Whether those ticked can be saved: the server refuses none of them. */
  readonly readiness = output<boolean>();
  readonly list = signal<LayerOps | null>(null);
  /**
   * Whether the list is unfolded, once the user folded or unfolded it; until then, whether any
   * operator is ticked.
   */
  readonly expanded = signal<boolean | null>(null);
  readonly error = signal('');
  private readonly api = inject(ArtifactsApi);
  private destroyed = false;
  private request = 0;
  constructor() {
    effect(() => void this.load(this.ref(), this.target(), this.selected()));
  }
  ngOnDestroy(): void {
    this.destroyed = true;
  }
  ticked(id: string): boolean {
    return this.selected().includes(id);
  }
  toggle(id: string, on: boolean): void {
    const order = this.list()?.blocks.flatMap((block) => block.ops.map((op) => op.id)) ?? [];
    const chosen = new Set(this.selected());
    if (on) {
      chosen.add(id);
    } else {
      chosen.delete(id);
    }
    this.selectionChange.emit(order.filter((op) => chosen.has(op)));
  }
  private async load(ref: string, target: string, ops: string[]): Promise<void> {
    const request = ++this.request;
    this.error.set('');
    if (!ref || !target) {
      this.list.set(null);
      this.readiness.emit(true);
      return;
    }
    try {
      const found = await this.api.layerOps(ref, target, ops);
      if (request !== this.request || this.destroyed) {
        return;
      }
      this.list.set(found);
      this.readiness.emit(!found.refusal);
    } catch (error) {
      if (request === this.request && !this.destroyed) {
        this.list.set(null);
        this.error.set(error instanceof Error ? error.message : String(error));
        this.readiness.emit(!ops.length);
      }
    }
  }
}
