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

import {ExplorerStatisticTable} from '../../shared/explorer_statistic_table/explorer_statistic_table';
import {MatIconModule} from '@angular/material/icon';
import {Component, computed, input, signal} from '@angular/core';
import type {SamplingDistribution, SamplingProbability} from '../../data/contracts/token_analysis';
import {ExplorerInfoValue} from '../../shared/explorer_info_value/explorer_info_value';
import {formatAnalysisMetric} from '../../shared/format/format';
import {runtimeSelectionBackground} from './runtime_series';

type Side = 'ref' | 'target';
interface CandidateRow {
  key: string;
  id: number;
  token: string | null;
  ref: SamplingProbability | null;
  target: SamplingProbability | null;
}
export function samplingTopCandidates(
  source: SamplingDistribution | undefined,
  side: Side,
  k: number,
) {
  return (source?.rows ?? [])
    .flatMap((row) => {
      const value = row[side];
      return value && Number.isFinite(value.probability) && value.probability > 0
        ? [{id: row.id, token: row.label, ...value}]
        : [];
    })
    .sort((a, b) => b.probability - a.probability || a.id - b.id)
    .slice(0, k);
}

/** Union each runtime's Top 64 by token ID, retaining at most 128 candidates. */
export function samplingCandidateUnion(
  ref: SamplingDistribution | undefined,
  target: SamplingDistribution | undefined,
): CandidateRow[] {
  const compatible =
    (!ref || ref.vocabulary_compatible) && (!target || target.vocabulary_compatible);
  const sources = {ref, target};
  const maps = {
    ref: new Map((ref?.rows ?? []).map((row) => [row.id, row.ref])),
    target: new Map((target?.rows ?? []).map((row) => [row.id, row.target])),
  };
  const probability = (side: Side, id: number): SamplingProbability | null => {
    const value = maps[side].get(id);
    if (value) return Number.isFinite(value.probability) ? value : null;
    const source = sources[side];
    // An absent entry is zero only when available sparse support is complete.
    return source?.rows_complete &&
      source[side].candidate_count != null &&
      Number.isFinite(source[side].entropy)
      ? {probability: 0, rank: null, selected: false}
      : null;
  };
  const rows = new Map<string, CandidateRow>();
  for (const side of ['ref', 'target'] as const) {
    for (const row of samplingTopCandidates(sources[side], side, 64)) {
      const key = compatible ? String(row.id) : `${side}:${row.id}`;
      if (!rows.has(key))
        rows.set(key, {
          key,
          id: row.id,
          token: row.token,
          ref: compatible || side === 'ref' ? probability('ref', row.id) : null,
          target: compatible || side === 'target' ? probability('target', row.id) : null,
        });
      else if (rows.get(key)!.token === null) rows.get(key)!.token = row.token;
    }
  }
  return [...rows.values()];
}

@Component({
  selector: 'sampling-distribution-inspection',
  imports: [ExplorerInfoValue, MatIconModule, ExplorerStatisticTable],
  template: `
    <table
      explorer-statistic-table
      class="layer-table"
      label="Sampling distribution probabilities"
      [columnCount]="4"
      [header]="header"
      [row]="row"
      [rows]="tableRows()"
    ></table>
    <ng-template #header>
      <tr>
        <th scope="col">
          <explorer-info-value
            [inline]="true"
            [textAction]="true"
            label="Token ID column"
            value="Token ID"
          />
        </th>
        @for (side of sides; track side) {
          <th scope="col" [attr.aria-sort]="sort().side === side ? sort().direction : 'none'">
            <div class="column-heading">
              <explorer-info-value
                [inline]="true"
                [textAction]="true"
                [label]="(side === 'ref' ? 'Ref Prob' : 'Target Prob') + ' column'"
                [value]="side === 'ref' ? 'Ref Prob' : 'Target Prob'"
              />
              <button
                class="column-sort"
                type="button"
                [disabled]="candidates().length < 2"
                [attr.aria-label]="'Sort by ' + (side === 'ref' ? 'Ref Prob' : 'Target Prob')"
                [title]="'Sort by ' + (side === 'ref' ? 'Ref Prob' : 'Target Prob')"
                (click)="sortBy(side)"
              >
                <mat-icon aria-hidden="true">{{
                  sort().side !== side
                    ? 'swap_vert'
                    : sort().direction === 'descending'
                      ? 'arrow_downward'
                      : 'arrow_upward'
                }}</mat-icon>
              </button>
            </div>
          </th>
        }
        <th scope="col" colspan="2">
          <explorer-info-value
            [inline]="true"
            [textAction]="true"
            label="Token column"
            value="Token"
          />
        </th>
      </tr>
    </ng-template>
    <ng-template #row let-item>
      <tr [attr.data-token-id]="item.candidate?.id">
        <td>{{ item.candidate?.id ?? '-' }}</td>
        @for (side of sides; track side) {
          <td
            [class.selected-cell]="item.candidate?.[side]?.selected"
            [style.background-color]="
              item.candidate?.[side]?.selected ? selectionBackground(side) : null
            "
          >
            <span class="stat-cell">
              <explorer-info-value
                class="stat-value"
                [inline]="true"
                [label]="side + ' probability for token ' + (item.candidate?.id ?? '')"
                [value]="probability(item.candidate?.[side]?.probability)"
              />
              <span class="stat-locator" aria-hidden="true"></span>
            </span>
          </td>
        }
        <td colspan="2">
          <explorer-info-value
            [inline]="true"
            [label]="'Token ' + (item.candidate?.id ?? '')"
            [value]="item.candidate ? tokenText(item.candidate.token) : '-'"
          />
        </td>
      </tr>
    </ng-template>
  `,
  styleUrl: './sampling_distribution_inspection.scss',
})
export class SamplingDistributionInspection {
  readonly refDistribution = input<SamplingDistribution>();
  readonly targetDistribution = input<SamplingDistribution>();
  readonly singlePosition = input(false);
  readonly sides = ['ref', 'target'] as const;
  readonly selectionBackground = runtimeSelectionBackground;
  readonly candidates = computed(() =>
    this.singlePosition()
      ? samplingCandidateUnion(this.refDistribution(), this.targetDistribution())
      : [],
  );
  readonly sort = signal<{side: Side; direction: 'ascending' | 'descending'}>({
    side: 'ref',
    direction: 'descending',
  });
  readonly sortedCandidates = computed(() => {
    const {side, direction} = this.sort();
    return [...this.candidates()].sort((a, b) => {
      const av = a[side]?.probability,
        bv = b[side]?.probability;
      if (av == null || bv == null) return av == null ? (bv == null ? a.id - b.id : 1) : -1;
      return (av - bv) * (direction === 'ascending' ? 1 : -1) || a.id - b.id;
    });
  });
  readonly tableRows = computed<Array<{label: string; candidate: CandidateRow | null}>>(() =>
    this.sortedCandidates().length
      ? this.sortedCandidates().map((candidate) => ({label: candidate.key, candidate}))
      : [{label: 'empty', candidate: null}],
  );
  sortBy(side: Side) {
    this.sort.update((current) => ({
      side,
      direction:
        current.side === side && current.direction === 'descending' ? 'ascending' : 'descending',
    }));
  }
  probability(value: number | undefined) {
    return value === undefined || !Number.isFinite(value)
      ? '-'
      : formatAnalysisMetric('selected_probability', value);
  }
  tokenText(value: string | null) {
    return value === null ? '-' : JSON.stringify(value);
  }
}
