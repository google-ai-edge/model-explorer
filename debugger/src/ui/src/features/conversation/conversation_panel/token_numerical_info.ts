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

import {Component, input} from '@angular/core';
import {MatTooltipModule} from '@angular/material/tooltip';
import {TokenAnalysisPair} from '../../../data/contracts/token_analysis';
import {ExplorerInfoValue} from '../../../shared/explorer_info_value/explorer_info_value';
import {heatLevel, heatPalette, metricText} from '../token_analysis';
import {TOKEN_METRICS} from '../token_metric_metadata';
@Component({
  selector: 'token-numerical-info',
  imports: [MatTooltipModule, ExplorerInfoValue],
  template: `
    @for (group of groups; track group.label) {
      <section>
        <h3>{{ group.label }}</h3>
        <table [attr.aria-label]="group.label + ' numerical differences'">
          <tbody>
            @for (metric of group.metrics; track metric.key) {
              <tr [attr.data-analysis-metric]="metric.key">
                <th>
                  <explorer-info-value
                    [label]="metric.label + ' — ' + metric.unit"
                    [value]="label(metric.label)"
                  />
                </th>
                <td>
                  <span
                    tabindex="0"
                    [attr.aria-label]="
                      metric.label +
                      ': ' +
                      value(metric.key) +
                      '; ' +
                      metric.unit +
                      '; ' +
                      reason(metric.key)
                    "
                    [matTooltip]="
                      metric.description + ' ' + metric.unit + '. ' + reason(metric.key)
                    "
                    [class.numeric-token]="analysis()?.metrics?.[metric.key]?.value != null"
                    [attr.data-palette]="heatPalette(metric.key)"
                    [style.--heat-level]="heatLevel(metric.key, analysis())"
                    >{{ displayValue(metric.key) }}</span
                  >
                </td>
              </tr>
            }
          </tbody>
        </table>
      </section>
    }
    @if (!analysis()) {
      <p class="info-note">
        {{
          loading()
            ? 'Loading captured token metrics…'
            : error()
              ? 'Numerical evidence unavailable.'
              : 'Per-token numerical evidence not captured.'
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
    @include token.numeric-colors;
    h3 {
      margin: 6px 12px 2px var(--info-indent, 12px);
      font: 500 12px/20px var(--me-font-family);
      color: var(--me-on-surface-color);
    }
    th:first-child {
      width: 176px;
      box-sizing: border-box;
    }
    th:first-child::before {
      content: '• ';
      color: var(--me-on-surface-variant-color);
    }
    th:first-child explorer-info-value {
      display: inline-flex;
      max-width: calc(100% - 12px);
      vertical-align: top;
    }
    td {
      text-align: left;
    }
    td span {
      display: inline-block;
      vertical-align: top;
      max-width: 100%;
      padding: 0 3px;
      box-sizing: border-box;
      overflow: hidden;
      text-overflow: ellipsis;
      border-radius: 2px;
      font-variant-numeric: tabular-nums;
    }
    td span:focus-visible {
      outline: 1px solid var(--me-primary-color);
      outline-offset: -1px;
    }
  `,
})
export class TokenNumericalInfo {
  readonly analysis = input<TokenAnalysisPair>();
  readonly loading = input(false);
  readonly error = input('');
  readonly heatLevel = heatLevel;
  readonly heatPalette = heatPalette;
  readonly groups = [
    {label: 'Logits', metrics: TOKEN_METRICS.slice(0, 2)},
    {label: 'Activations', metrics: TOKEN_METRICS.slice(2)},
  ];
  label(value: string) {
    return value.split(' · ')[1];
  }
  value(key: (typeof TOKEN_METRICS)[number]['key']) {
    return !this.analysis() && this.loading()
      ? 'Loading…'
      : !this.analysis() && this.error()
        ? 'Unavailable'
        : metricText(this.analysis()?.metrics[key]);
  }
  /** A value with its unit; a missing value reads as a dash, its reason stays in the tooltip. */
  displayValue(key: (typeof TOKEN_METRICS)[number]['key']) {
    const value = this.analysis()?.metrics[key]?.value;
    const units: Record<string, string> = {
      kl: ' nats',
      js: ' bits',
      relative_l2: ' %',
      norm_ratio: ' ×',
    };
    if (value == null || !Number.isFinite(value))
      return this.loading() && !this.analysis() ? 'Loading…' : '—';
    return metricText(this.analysis()?.metrics[key], key) + (units[key] ?? '');
  }
  reason(key: (typeof TOKEN_METRICS)[number]['key']) {
    return this.analysis()?.metrics[key]?.reason ?? '';
  }
}
