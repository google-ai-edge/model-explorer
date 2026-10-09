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
  inject,
  input,
  OnDestroy,
  OnInit,
  output,
  signal,
} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {ReportApiService} from '../../../data/report_api_service';

import type {TapPoint, TapScan} from '../../../data/contracts/tap_scan';

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
  readonly preset = input(false);
  readonly selected = input<string[]>([]);
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
    const words = this.query()
      .toLowerCase()
      .trim()
      .split(/\s+/)
      .filter(Boolean);
    return (this.scan().points ?? []).filter(
      (p) =>
        (!this.signature() || p.signature === this.signature()) &&
        words.every((w) =>
          `${p.tensor_name} ${p.operator} ${p.op} ${p.signature}`
            .toLowerCase()
            .includes(w),
        ),
    );
  });
  readonly visible = computed(() => this.matches().slice(0, 80));
  private api = inject(ReportApiService);
  private destroyed = false;
  ngOnInit() {
    void this.start();
  }
  ngOnDestroy() {
    this.destroyed = true;
  }
  async start() {
    this.scan.set({status: 'scanning'});
    this.readiness.emit(false);
    try {
      let result = await this.api.post<TapScan>(
        `artifacts/${this.artifact()}/scan`,
        {},
      );
      while (result.status === 'scanning') {
        await new Promise((resolve) => setTimeout(resolve, 750));
        if (this.destroyed) return;
        result = await this.api.tapScan(this.artifact());
      }
      if (this.destroyed) return;
      this.scan.set(result);
      this.readiness.emit(result.status === 'completed');
      if (
        result.status === 'completed' &&
        this.preset() &&
        !this.selected().length &&
        result.recommended?.length
      )
        this.selectionChange.emit(result.recommended);
    } catch (error) {
      if (!this.destroyed)
        this.scan.set({status: 'failed', error: (error as Error).message});
    }
  }
  labelFor(point: TapPoint) {
    const layer = point.tensor_name
      .match(/(?:^|\/)layer_\d+(?=\/)/)?.[0]
      .replaceAll('/', '');
    const leaf = point.tensor_name.split('/').slice(-2).join('/');
    return [layer, leaf || `Tensor ${point.tensor}`]
      .filter(Boolean)
      .join(' / ');
  }
  toggle(point: TapPoint) {
    if (this.disabled() || !point.selectable) return;
    const selected = this.selected();
    this.selectionChange.emit(
      selected.includes(point.id)
        ? selected.filter((id) => id !== point.id)
        : selected.length < 16
          ? [...selected, point.id]
          : selected,
    );
  }
  recommended() {
    this.selectionChange.emit(this.scan().recommended ?? []);
  }
}
