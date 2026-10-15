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

/** Input/metadata presentation adapted from Model Explorer info_panel (Apache-2.0). */
import {Component, input, output} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {ExplorerLocator} from '../explorer_locator/explorer_locator';
import {ExplorerInfoValue} from '../explorer_info_value/explorer_info_value';

import type {ExplorerValueSegment} from '../explorer_info_value/explorer_help_term';

export interface ExplorerInfoItem {
  segments?: readonly ExplorerValueSegment[];
  label: string;
  value?: string;
  locateLabel?: string;
  expandable?: boolean;
  wrap?: boolean;
  children?: {label: string; value: string}[];
}

@Component({
  selector: 'explorer-info-list',
  imports: [MatIconModule, ExplorerLocator, ExplorerInfoValue],
  host: {'[class.indexed]': 'indexed()'},
  template: `
    <div class="flat-items-container" role="list">
      @for (item of items(); track item.label; let i = $index) {
        <div class="flat-item" role="listitem" [attr.aria-label]="item.label">
          @if (item.children?.length || (indexed() && item.value === undefined)) {
            <div class="name-row">
              @if (indexed()) {
                <span class="item-index">{{ i }}</span>
              }
              @if (item.locateLabel) {
                <button
                  explorer-locator
                  [label]="item.label"
                  [attr.aria-label]="item.locateLabel"
                  [attr.title]="item.locateLabel"
                  (click)="locate.emit(item)"
                ></button>
              } @else {
                <span class="item-name">{{ item.label }}</span>
              }
            </div>
          } @else {
            <table class="metadata-table" [attr.aria-label]="item.label + ' information'">
              <tbody>
                <tr [class.wrap-label]="item.wrap">
                  <th [attr.colspan]="item.value === undefined ? 2 : null">
                    <div class="key-container">
                      @if (bullets()) {
                        <mat-icon class="bullet" aria-hidden="true">circle</mat-icon>
                      }
                      {{ item.label }}
                    </div>
                  </th>
                  @if (item.value !== undefined) {
                    <td class="value">
                      @if (item.expandable || item.locateLabel || item.segments) {
                        <div class="property-value">
                          <explorer-info-value
                            [value]="item.value"
                            [segments]="item.segments ?? []"
                            [label]="item.label"
                            [inline]="true"
                            [wrap]="item.wrap ?? false"
                          />
                          @if (item.locateLabel) {
                            <button
                              explorer-locator
                              [attr.aria-label]="item.locateLabel"
                              [title]="item.locateLabel"
                              (click)="locate.emit(item)"
                            ></button>
                          }
                        </div>
                      } @else {
                        {{ item.value }}
                      }
                    </td>
                  }
                </tr>
              </tbody>
            </table>
          }
          @if (item.children?.length) {
            <table
              class="metadata-table child-properties"
              [attr.aria-label]="item.label + ' information'"
            >
              <tbody>
                @for (child of item.children; track child.label) {
                  <tr>
                    <th>
                      <div class="key-container">
                        <mat-icon class="bullet" aria-hidden="true">circle</mat-icon
                        >{{ child.label }}
                      </div>
                    </th>
                    <td>
                      <explorer-info-value
                        [value]="child.value"
                        [label]="item.label + ' ' + child.label"
                        [inline]="true"
                      />
                    </td>
                  </tr>
                }
              </tbody>
            </table>
          }
        </div>
      }
    </div>
  `,
  styles: `
    @use '../../theme/explorer_info' as info;
    @include info.metadata;
    :host {
      display: block;
      min-width: 0;
      margin: 0 16px 0 var(--info-indent, 12px);
      font: 12px / normal var(--me-font-family);
    }
    // Bullets express nesting; shared table widths keep every value column aligned.
    .flat-items-container {
      display: flex;
      flex-direction: column;
      gap: var(--info-property-gap, 4px);
    }
    .flat-item {
      display: flex;
      flex-direction: column;
    }
    .name-row {
      min-height: var(--info-name-row-height, 24px);
      display: flex;
      align-items: center;
      gap: 2px;
      color: var(--me-on-surface-color);
    }
    .item-name {
      color: var(--me-primary-color);
      font-weight: 500;
    }
    .item-index {
      min-width: var(--info-index-width, 14px);
      text-align: var(--info-index-align, left);
      color: var(--me-on-surface-variant-color);
    }
    .metadata-table.child-properties {
      margin-top: var(--info-child-spacing, 0px);
      margin-left: var(--info-child-indent, #{info.$child-indent});
      width: calc(100% - var(--info-child-indent, #{info.$child-indent}));
    }
    .name-row {
      padding: var(--info-name-row-padding, 1px) 0;
      font-weight: 400;
      color: var(--me-on-surface-low-color);
    }
    .metadata-table {
      margin: 0;
      width: 100%;
      max-width: 100%;
    }
    th:first-child {
      width: calc(var(--info-key-width, 100px) - var(--info-key-offset, 0px));
    }
    th,
    td {
      vertical-align: top;
    }
    .wrap-label th {
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .property-value {
      display: flex;
      align-items: flex-start;
      gap: 2px;
      min-width: 0;
    }
    .property-value explorer-info-value {
      flex: 1;
    }
    .property-value button[explorer-locator] {
      height: 16px;
    }
    .key-container {
      display: flex;
      align-items: flex-start;
    }
    mat-icon.bullet {
      flex-shrink: 0;
      font-size: 4px;
      width: info.$bullet-width;
      height: 5px;
      color: var(--me-on-surface-low-color);
      margin-right: info.$bullet-gap;
      margin-top: 6px;
    }
    td {
      white-space: normal;
      overflow-wrap: anywhere;
    }
  `,
})
export class ExplorerInfoList {
  readonly locate = output<ExplorerInfoItem>();
  readonly indexed = input(false);
  readonly bullets = input(false);
  readonly items = input.required<ExplorerInfoItem[]>();
}
