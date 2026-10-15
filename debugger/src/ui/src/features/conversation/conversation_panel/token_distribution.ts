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

import {Component, computed, input, linkedSignal, signal} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import type {
  SamplingDistributionRow,
  TokenAnalysisPair,
} from '../../../data/contracts/token_analysis';
import {ExplorerInfoSection} from '../../../shared/explorer_info_section/explorer_info_section';
import {ExplorerInfoValue} from '../../../shared/explorer_info_value/explorer_info_value';
import type {ExplorerValueSegment} from '../../../shared/explorer_info_value/explorer_help_term';
import {ExplorerPagination} from '../../../shared/explorer_pagination/explorer_pagination';
import {formatAnalysisMetric} from '../../../shared/format/format';

export type SamplingCandidateColumn = 'id' | 'ref' | 'target';
type SortDirection = 'ascending' | 'descending';

/** Inspect one position's sampling evidence; never substitute the raw-logit softmax. */
@Component({
  selector: 'token-distribution',
  imports: [MatIconModule, ExplorerInfoSection, ExplorerInfoValue, ExplorerPagination],
  templateUrl: './token_distribution.ng.html',
  styleUrl: './token_distribution.scss',
})
export class TokenDistribution {
  readonly analysis = input<TokenAnalysisPair>();
  readonly singlePosition = input(true);
  readonly loading = input(false);
  readonly error = input('');
  readonly distribution = computed(() => this.analysis()?.sampling_distribution);
  readonly sides = [
    {key: 'ref', label: 'Reference', short: 'Ref'},
    {key: 'target', label: 'Target', short: 'Target'},
  ] as const;
  readonly columns = [
    {key: 'id', label: 'Token ID', title: 'Token ID'},
    {key: 'ref', label: 'Ref', title: 'Reference probability'},
    {key: 'target', label: 'Target', title: 'Target probability'},
  ] as const;
  readonly summaryFields = [
    {key: 'selected_id', label: 'Selected token'},
    {key: 'candidate_count', label: 'Candidate count'},
    {key: 'selected_rank', label: 'Selected rank'},
  ] as const;
  readonly hasDistribution = computed(() =>
    this.sides.some(({key}) => Number.isFinite(this.distribution()?.[key].entropy)),
  );
  readonly unavailable = computed(() =>
    this.loading()
      ? 'Loading sampling distribution…'
      : this.error()
        ? 'Sampling distribution unavailable: ' + this.error()
        : (this.distribution()?.reason ??
          'Sampling distribution unavailable. Sampling evidence was not captured.'),
  );
  readonly sort = signal<{column: SamplingCandidateColumn; direction: SortDirection}>({
    column: 'id',
    direction: 'ascending',
  });
  readonly sortedRows = computed(() => {
    const {column, direction} = this.sort();
    const value = (row: SamplingDistributionRow) =>
      column === 'id' ? row.id : row[column]?.probability;
    return [...(this.distribution()?.rows ?? [])].sort((a, b) => {
      const av = value(a),
        bv = value(b);
      const aValid = typeof av === 'number' && Number.isFinite(av);
      const bValid = typeof bv === 'number' && Number.isFinite(bv);
      return (
        Number(bValid) - Number(aValid) ||
        (aValid && bValid ? (av! - bv!) * (direction === 'ascending' ? 1 : -1) : 0) ||
        a.id - b.id
      );
    });
  });
  readonly pageSize = 16;
  readonly pageIndex = linkedSignal({source: () => this.analysis(), computation: () => 0});
  readonly pages = computed(() => Math.max(1, Math.ceil(this.sortedRows().length / this.pageSize)));
  readonly currentPage = computed(() => Math.min(this.pageIndex(), this.pages() - 1));
  readonly rows = computed(() => {
    const start = this.currentPage() * this.pageSize;
    return this.sortedRows().slice(start, start + this.pageSize);
  });
  readonly identityWidth = linkedSignal({
    source: () => ({rows: this.rows(), pending: this.loading() || !!this.error()}),
    computation: (current, previous) => {
      if (current.pending && !current.rows.length) return previous?.value ?? 64;
      return Math.max(
        64,
        Math.ceil(Math.max(0, ...current.rows.map((row) => String(row.id).length)) * 6.75 + 14),
      );
    },
  });
  columnSortDirection(column: SamplingCandidateColumn) {
    return this.sort().column === column ? this.sort().direction : 'none';
  }
  sortBy(column: SamplingCandidateColumn) {
    this.sort.set({
      column,
      direction: this.columnSortDirection(column) === 'ascending' ? 'descending' : 'ascending',
    });
    this.pageIndex.set(0);
  }
  changePage(delta: number) {
    this.pageIndex.set(Math.max(0, Math.min(this.pages() - 1, this.currentPage() + delta)));
  }
  summaryValue(side: 'ref' | 'target', key: (typeof this.summaryFields)[number]['key']) {
    const value = this.distribution()?.[side][key];
    if (value == null || !Number.isFinite(value)) return '—';
    if (key === 'selected_rank') return '#' + value;
    return String(value);
  }
  summarySegments(side: 'ref' | 'target', key: (typeof this.summaryFields)[number]['key']) {
    if (key !== 'selected_id') return [];
    const distribution = this.distribution();
    const id = distribution?.[side].selected_id;
    if (id == null) return [];
    const candidate = distribution?.rows.find((row) => row.id === id);
    const compatible = candidate?.[side] != null || distribution?.vocabulary_compatible === true;
    return this.tokenSegments({id, label: compatible ? (candidate?.label ?? null) : null});
  }
  tokenSegments(row: Pick<SamplingDistributionRow, 'id' | 'label'>): ExplorerValueSegment[] {
    return [
      {
        text: String(row.id),
        help: {
          title: 'Token',
          text: row.label === null ? 'Text not captured' : row.label,
          rows: [],
        },
      },
    ];
  }
  probability(value: SamplingDistributionRow['ref']) {
    return formatAnalysisMetric('selected_probability', value?.probability);
  }
  probabilitySegments(
    row: SamplingDistributionRow,
    side: 'ref' | 'target',
  ): ExplorerValueSegment[] {
    const value = row[side];
    if (!value) return [];
    return [
      {
        text: this.probability(value),
        help: {
          title: `${side === 'ref' ? 'Reference' : 'Target'} · Token ${row.id}`,
          rows: [
            {
              label: 'Probability',
              value: Number.isFinite(value.probability) ? String(value.probability) : 'Unavailable',
            },
            {
              label: 'Rank',
              value: value.rank == null ? 'Outside sampling support' : '#' + value.rank,
            },
            {label: 'Generated', value: value.selected ? 'Yes · selected token' : 'No'},
          ],
        },
      },
    ];
  }
}
