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

import {Component, computed, input} from '@angular/core';
import {MatTooltipModule} from '@angular/material/tooltip';
import {
  TokenAnalysisPair,
  TokenProbability,
} from '../../../data/contracts/token_analysis';
import {formatFixed} from '../../../shared/format/format';
import {spokenTokenText, tokenMarkup} from '../conversation_content';
/** Candidate ordering, pinned selected IDs and pp units follow candidates HTML. */
@Component({
  selector: 'token-distribution',
  imports: [MatTooltipModule],
  template: `
    @if (hasDistribution()) {
      <h3>Summary</h3>
      <table aria-label="Full-vocabulary distribution summary">
        <colgroup>
          <col class="property" />
          <col />
          <col />
          <col class="delta" />
        </colgroup>
        <thead>
          <tr>
            <th></th>
            <th>Reference</th>
            <th>Target</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th>Entropy (bits)</th>
            <td>{{ entropy('ref') }}</td>
            <td>{{ entropy('target') }}</td>
            <td></td>
          </tr>
          <tr>
            <th>Selected rank</th>
            <td>{{ rank('ref') }}</td>
            <td>{{ rank('target') }}</td>
            <td></td>
          </tr>
          <tr>
            <th>Selected probability</th>
            <td>{{ share('ref', 'selected_probability') }}</td>
            <td>{{ share('target', 'selected_probability') }}</td>
            <td></td>
          </tr>
          <tr>
            <th
              matTooltip="Probability of the most likely token minus the second most likely, in percentage points: how close this step was to choosing differently."
              matTooltipPosition="above"
            >
              Top-2 margin
            </th>
            <td>{{ share('ref', 'margin', ' pp') }}</td>
            <td>{{ share('target', 'margin', ' pp') }}</td>
            <td></td>
          </tr>
        </tbody>
      </table>
      <div class="candidate-heading">
        <h3>Candidates</h3>
        <span>{{ distribution()?.rows?.length ?? 0 }} tokens</span>
      </div>
      @if (distribution()?.context === 'different') {
        <p class="info-note context-note" [matTooltip]="distribution()?.reason ?? ''">
          Different contexts · each runtime's own candidates, not comparable
        </p>
      }
      @if (distribution()?.rows?.length) {
        <table
          class="probability-table"
          aria-label="Model token probabilities; highlighted cells indicate selected tokens"
        >
          <colgroup>
            <col class="property" />
            <col />
            <col />
            <col class="delta" />
          </colgroup>
          <thead>
            <tr>
              <th>Token</th>
              <th>Reference</th>
              <th>Target</th>
              @if (distribution()?.compatible) {
                <th>
                  <span
                    tabindex="0"
                    matTooltip="Target minus Reference probability, in percentage points"
                    >ΔP (pp)</span
                  >
                </th>
              }
            </tr>
          </thead>
          <tbody>
            @for (row of distribution()!.rows; track row.id) {
              <tr [attr.data-candidate-id]="row.id">
                <th>
                  <span
                    class="candidate-token"
                    tabindex="0"
                    [attr.title]="
                      'Token ID ' + row.id + (row.label === null ? ' · Text not captured' : '')
                    "
                    [attr.aria-label]="
                      'Token ID ' +
                      row.id +
                      ': ' +
                      (row.label === null ? 'text not captured' : spoken(row.label))
                    "
                    [innerHTML]="row.label === null ? '#' + row.id : markup(row.label)"
                  ></span>
                </th>
                @for (side of sides; track side) {
                  <td>
                    <span
                      tabindex="0"
                      [attr.data-selected-side]="row[side]?.selected ? side : null"
                      [attr.aria-label]="
                        (row[side]?.selected ? 'Selected token probability ' : '') +
                        probability(row[side])
                      "
                      [matTooltip]="tooltip(row[side])"
                      >{{ probability(row[side]) }}</span
                    >
                  </td>
                }
                @if (distribution()?.compatible) {
                  <td>{{ delta(row.delta) }}</td>
                }
              </tr>
            }
          </tbody>
        </table>
      }
      @if (distribution()?.reason && distribution()?.context !== 'different') {
        <p class="info-note">{{ distribution()?.reason }}</p>
      }
    } @else {
      <p class="info-note">
        {{
          loading()
            ? 'Loading captured distribution…'
            : error()
              ? 'Distribution unavailable.'
              : 'Full-vocabulary logits not captured.'
        }}
      </p>
    }
  `,
  styles: `
    @use '../../../theme/explorer_info' as info;
    @use '../../../theme/token_diff' as token;
    :host {
      display: block;
      @include token.semantic-colors;
    }
    @include info.metadata;
    table {
      margin-left: calc(var(--info-indent, 12px) + 12px);
      width: calc(100% - 28px - var(--info-indent, 12px));
      max-width: calc(100% - 28px - var(--info-indent, 12px));
    }
    @include info.comparison-grid;
    h3 {
      margin: 6px 12px 2px var(--info-indent, 12px);
      font: 500 12px/20px var(--me-font-family);
      color: var(--me-on-surface-color);
    }
    table {
      table-layout: fixed;
    }
    th:first-child {
      padding-right: 4px;
    }
    td {
      font-variant-numeric: tabular-nums;
    }
    td span {
      display: inline-block;
      vertical-align: top;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      padding: 0 2px;
      box-sizing: border-box;
      border-radius: 2px;
    }
    .candidate-token {
      display: inline-block;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: pre;
      vertical-align: top;
      color: var(--me-on-surface-color);
    }
    .candidate-heading {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding-right: 16px;
    }
    .candidate-heading > span {
      font: 11px/20px var(--me-font-family);
      color: var(--me-on-surface-variant-color);
    }
    [data-selected-side='ref'] {
      background: var(--b0);
    }
    [data-selected-side='target'] {
      background: var(--b4);
    }
    span:focus-visible {
      outline: 1px solid var(--me-primary-color);
      outline-offset: -1px;
    }
  `,
})
export class TokenDistribution {
  readonly analysis = input<TokenAnalysisPair>();
  readonly loading = input(false);
  readonly error = input('');
  readonly distribution = computed(() => this.analysis()?.distribution);
  readonly hasDistribution = computed(
    () =>
      this.distribution()?.ref.entropy !== undefined ||
      this.distribution()?.target.entropy !== undefined,
  );
  readonly sides = ['ref', 'target'] as const;
  readonly markup = tokenMarkup;
  readonly spoken = spokenTokenText;
  entropy(side: 'ref' | 'target') {
    return formatFixed(this.distribution()?.[side].entropy, 3);
  }
  /** A probability of one side's own distribution, as a percentage or in percentage points. */
  share(
    side: 'ref' | 'target',
    key: 'selected_probability' | 'margin',
    unit = '%',
  ) {
    const value = this.distribution()?.[side][key];
    return value == null || !Number.isFinite(value)
      ? '—'
      : (value * 100).toFixed(1) + unit;
  }
  rank(side: 'ref' | 'target') {
    const v = this.distribution()?.[side].selected_rank;
    return v == null ? '—' : '#' + v;
  }
  // Preserve candidates HTML formatTokenScore's display thresholds.
  probability(value: TokenProbability | null) {
    const p = value?.probability;
    if (p == null || !Number.isFinite(p)) return '—';
    if (p === 0) return '0%';
    if (p < 0.001) return '<0.1%';
    return (p * 100).toFixed(1) + '%';
  }
  delta(value: number | null) {
    return value === null
      ? '—'
      : (value > 0 ? '+' : '') + formatFixed(value, 2);
  }
  tooltip(value: TokenProbability | null) {
    return value
      ? 'Logit ' + value.logit + ' · Rank #' + value.rank
      : 'Probability not captured';
  }
}
