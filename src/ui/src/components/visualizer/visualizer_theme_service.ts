/**
 * @license
 * Copyright 2024 The Model Explorer Authors. All Rights Reserved.
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

import {AppService} from './app_service';

import {ElementRef, Injectable, inject} from '@angular/core';

/**
 * Color variables that can be used to style the visualizer.
 */
export enum ColorVariable {
  PRIMARY_COLOR = '--me-primary-color',
  SURFACE_CONTAINER_HIGHEST_COLOR = '--me-surface-container-highest-color',
  SURFACE_CONTAINER_HIGH_COLOR = '--me-surface-container-high-color',
  SURFACE_CONTAINER_COLOR = '--me-surface-container-color',
  SURFACE_CONTAINER_LOW_COLOR = '--me-surface-container-low-color',
  SURFACE_CONTAINER_LOWEST_COLOR = '--me-surface-container-lowest-color',
  ON_SURFACE_COLOR = '--me-on-surface-color',
  ON_SURFACE_VARIANT_COLOR = '--me-on-surface-variant-color',
  ON_SURFACE_LOW_COLOR = '--me-on-surface-low-color',
  ON_SURFACE_INVERSE_COLOR = '--me-on-surface-inverse-color',
  SURFACE_COLOR = '--me-surface-color',
  SURFACE_VARIANT_COLOR = '--me-surface-variant-color',
  PRIMARY_CONTAINER_COLOR = '--me-primary-container-color',
  ON_PRIMARY_CONTAINER_COLOR = '--me-on-primary-container-color',
  SECONDARY_CONTAINER_COLOR = '--me-secondary-container-color',
  ON_SECONDARY_CONTAINER_COLOR = '--me-on-secondary-container-color',
  OUTLINE_COLOR = '--me-outline-color',
  OUTLINE_VARIANT_COLOR = '--me-outline-variant-color',
  OUTLINE_HAIRLINE_COLOR = '--me-outline-hairline-color',
  OUTLINE_HAIRLINE_VARIANT_COLOR = '--me-outline-hairline-variant-color',
  LINK_COLOR = '--me-link-color',
  DISABLED_BUTTON_BG_COLOR = '--me-disabled-button-bg-color',
  DISABLED_BUTTON_COLOR = '--me-disabled-button-color',
  SUCCESS_CONTAINER_COLOR = '--me-success-container-color',
  SUCCESS_TEXT_COLOR = '--me-success-text-color',
  WARNING_CONTAINER_COLOR = '--me-warning-container-color',
  WARNING_TEXT_COLOR = '--me-warning-text-color',
  ERROR_CONTAINER_COLOR = '--me-error-container-color',
  ERROR_TEXT_COLOR = '--me-error-text-color',
  TOOLTIP_BG_COLOR = '--me-tooltip-bg-color',
  BUBBLE_SHADOW = '--me-bubble-shadow',
  EDGE_COLOR = '--me-edge-color',
  EDGE_DIMMED_COLOR = '--me-edge-dimmed-color',
  GROUP_NODE_BG_COLOR1 = '--me-group-node-bg-color1',
  GROUP_NODE_BG_COLOR2 = '--me-group-node-bg-color2',
  GROUP_NODE_BG_COLOR3 = '--me-group-node-bg-color3',
  GROUP_NODE_BG_COLOR4 = '--me-group-node-bg-color4',
  GROUP_NODE_BG_COLOR5 = '--me-group-node-bg-color5',
  GROUP_NODE_BG_COLOR6 = '--me-group-node-bg-color6',
  INCOMING_EDGE_COLOR = '--me-incoming-edge-color',
  INCOMING_EDGE_TEXT_COLOR = '--me-incoming-edge-text-color',
  OUTGOING_EDGE_COLOR = '--me-outgoing-edge-color',
  OUTGOING_EDGE_TEXT_COLOR = '--me-outgoing-edge-text-color',
  IDENTICAL_GROUP_BG_COLOR = '--me-identical-group-bg-color',
  ARTIFICIAL_GROUPS_BORDER_COLOR = '--me-artificial-groups-border-color',
  SEARCH_RESULTS_HIGHLIGHT_COLOR = '--me-search-results-highlight-color',
  ACTIVE_PANEL_BAR_BG_COLOR = '--me-active-panel-bar-bg-color',
}

/**
 * Service to manage theme for the visualizer.
 */
@Injectable()
export class VisualizerThemeService {
  private root!: ElementRef<HTMLElement>;
  private readonly colorCache: Record<string, string> = {};

  private readonly appService = inject(AppService);

  init(root: ElementRef<HTMLElement>) {
    this.root = root;
  }

  getColor(variable: ColorVariable): string {
    const cacheKey = `${this.appService.theme()}__${variable}`;
    const cachedColor = this.colorCache[cacheKey];
    if (cachedColor) {
      return cachedColor;
    }
    const color =
      window
        .getComputedStyle(this.root.nativeElement)
        .getPropertyValue(variable) ?? 'pink';
    this.colorCache[cacheKey] = color;
    return color;
  }
}
