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
  Component,
  computed,
  effect,
  inject,
  input,
  output,
  signal,
  OnInit,
  OnDestroy,
} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {ArtifactsApi} from '../../server/artifacts_api';

import type {TapCounterparts, TapPoint, TapScan} from '../../server/contracts/artifacts';

@Component({
  selector: 'tap-picker',
  imports: [FormsModule, MatButtonModule, MatIconModule],
  templateUrl: './tap_picker.ng.html',
  styleUrl: './tap_picker.scss',
})
export class TapPicker implements OnInit, OnDestroy {
  readonly artifact = input.required<string>();
  readonly label = input.required<string>();
  readonly enabled = input(false);
  readonly disabled = input(false);
  readonly selected = input<string[]>([]);
  /**
   * The PyTorch model of the other side, when it observes: the server then says how each point
   * chosen pairs with it.
   */
  readonly pairedWith = input('');
  readonly selectionChange = output<string[]>();
  readonly readiness = output<boolean>();
  readonly scan = signal<TapScan>({status: 'scanning'});
  readonly expanded = signal(false);
  readonly query = signal('');
  readonly signature = signal('');
  readonly signatures = computed(() => [
    ...new Set((this.scan().points ?? []).map((p) => p.signature)),
  ]);
  readonly chosen = computed(() =>
    (this.scan().points ?? []).filter((p) => this.selected().includes(p.id)),
  );
  readonly matches = computed(() => {
    const words = this.query().toLowerCase().trim().split(/\s+/).filter(Boolean);
    return (this.scan().points ?? []).filter(
      (p) =>
        (!this.signature() || p.signature === this.signature()) &&
        words.every((w) =>
          `${p.tensor_name} ${p.operator} ${p.op} ${p.signature}`.toLowerCase().includes(w),
        ),
    );
  });
  readonly visible = computed(() => this.matches().slice(0, 80));
  /** The most points of the model a Session observes, as the server bounds them. */
  readonly limit = computed(() => this.scan().max_points ?? 16);
  /** How the server pairs the points chosen with the PyTorch model's, or null. */
  readonly pairing = signal<TapCounterparts | null>(null);
  private readonly api = inject(ArtifactsApi);
  private destroyed = false;
  private pairingRequest = 0;

  constructor() {
    effect(() => {
      const pytorch = this.pairedWith();
      const ids = this.selected();
      const ready = this.scan().status === 'completed';
      void this.loadPairing(pytorch, ready ? ids : []);
    });
  }

  private async loadPairing(pytorch: string, ids: string[]): Promise<void> {
    const request = ++this.pairingRequest;
    if (!pytorch || !ids.length) {
      this.pairing.set(null);
      return;
    }
    try {
      const found = await this.api.tapCounterparts(this.artifact(), pytorch, ids);
      if (request === this.pairingRequest && !this.destroyed) {
        this.pairing.set(found);
      }
    } catch {
      if (request === this.pairingRequest && !this.destroyed) {
        this.pairing.set(null);
      }
    }
  }

  /** The server's words on how a point chosen pairs, and whether it pairs at all. */
  pairingOf(id: string) {
    return this.pairing()?.points.find((p) => p.id === id) ?? null;
  }

  ngOnInit(): void {
    void this.start();
  }

  ngOnDestroy(): void {
    this.destroyed = true;
  }

  async start(): Promise<void> {
    this.scan.set({status: 'scanning'});
    this.readiness.emit(false);
    try {
      let result = await this.api.startTapScan(this.artifact());
      while (result.status === 'scanning') {
        await new Promise((resolve) => setTimeout(resolve, 750));
        if (this.destroyed) {
          return;
        }
        result = await this.api.tapScan(this.artifact());
      }
      if (this.destroyed) {
        return;
      }
      this.scan.set(result);
      this.readiness.emit(result.status === 'completed');
    } catch (error) {
      if (!this.destroyed) {
        this.scan.set({
          status: 'failed',
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }
  }

  labelFor(point: TapPoint): string {
    const layer = point.tensor_name.match(/(?:^|\/)layer_\d+(?=\/)/)?.[0]?.replaceAll('/', '');
    const leaf = point.tensor_name.split('/').slice(-2).join('/');
    return [layer, leaf || `Tensor ${point.tensor}`].filter(Boolean).join(' / ');
  }

  toggle(point: TapPoint): void {
    if (this.disabled() || !point.selectable) {
      return;
    }
    const selected = this.selected();
    this.selectionChange.emit(
      selected.includes(point.id)
        ? selected.filter((id) => id !== point.id)
        : selected.length < this.limit()
          ? [...selected, point.id]
          : selected,
    );
  }

  recommended(): void {
    this.selectionChange.emit(this.scan().recommended ?? []);
  }
}
