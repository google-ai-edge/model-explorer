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

import {computed, signal} from '@angular/core';
import type {KvFindResponse, KvFindResult} from '../../../data/contracts/kv';
import {compileKvQuery} from './kv_query';

export const KV_FIND_PAGE = 40;

/** What the controller needs from the KV panel: the request state and the effects of a pick. */
export interface KvFindHost {
  data(): KvFindResponse | null;
  loading(): boolean;
  select(result: KvFindResult): void;
  setResults(show: boolean): void;
  openPopup(): void;
  closePopup(): void;
}

/**
 * Find-formula state and result navigation for the KV explorer. Results arrive
 * in pages of 40; navigating past the loaded page requests it and selects the
 * result when it lands.
 */
export class KvFindController {
  readonly retry = signal(0);
  readonly formula = signal('');
  readonly queryError = signal('');
  readonly showResults = signal(false);
  readonly resultPage = signal(0);
  readonly resultIndex = signal(-1);
  readonly resultTotal = signal(0);
  draftFormula = '';
  private requestedResult = -1;
  readonly results = computed(() => this.host.data()?.results ?? []);
  readonly resultPages = computed(() =>
    Math.max(1, Math.ceil(this.resultTotal() / KV_FIND_PAGE)),
  );
  constructor(private readonly host: KvFindHost) {}

  /** A page of results arrived for `page`; keep the page in range and honour a pending jump. */
  receive(data: KvFindResponse, page: number) {
    this.resultTotal.set(data.total);
    if (page > 0 && data.offset >= data.total) {
      this.resultPage.set(
        Math.max(0, Math.ceil(data.total / KV_FIND_PAGE) - 1),
      );
      return;
    }
    const index = this.requestedResult - data.offset;
    if (index >= 0 && index < data.results.length) {
      this.select(data.results[index], index);
      this.requestedResult = -1;
    }
  }
  /** The user picked a position by hand; it is no longer "result n". */
  reset() {
    this.requestedResult = -1;
    this.resultIndex.set(-1);
  }
  select(result: KvFindResult, indexInPage: number) {
    this.host.select(result);
    this.resultIndex.set((this.host.data()?.offset ?? 0) + indexInPage);
  }
  navigate(index: number) {
    if (index < 0 || index >= this.resultTotal() || this.host.loading()) return;
    const offset = this.host.data()?.offset ?? 0,
      result = this.results()[index - offset];
    if (result) {
      this.select(result, index - offset);
      return;
    }
    this.requestedResult = index;
    this.resultPage.set(Math.floor(index / KV_FIND_PAGE));
  }
  move(delta: number) {
    this.navigate(Math.max(0, this.resultIndex() + delta));
  }
  first() {
    this.navigate(0);
  }
  open() {
    this.draftFormula = this.formula();
    this.queryError.set('');
    this.host.openPopup();
  }
  apply() {
    try {
      compileKvQuery(this.draftFormula);
      this.formula.set(this.draftFormula.trim());
      this.host.setResults(!!this.draftFormula.trim());
      this.queryError.set('');
      this.host.closePopup();
    } catch (error) {
      this.queryError.set((error as Error).message);
    }
  }
  clear() {
    this.draftFormula = '';
    this.formula.set('');
    this.showResults.set(false);
    this.host.closePopup();
  }
}
