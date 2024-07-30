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

import {Injectable} from '@angular/core';

/** Sorting direction. */
export type SortingDirection = 'asc' | 'desc';

/**
 * A service scoped inside the info panel.
 */
@Injectable()
export class InfoPanelService {
  // -2: the index column (default).
  // -1: the node label column.
  curSortingRunIndex = -2;
  curSortingDirection: SortingDirection = 'asc';

  curChildrenStatSortingColIndex = -2;
  curChildrenStatSortingDirection: SortingDirection = 'asc';

  statsTableCollapsed = false;
  childrenStatsTableCollapsed = false;
  nodeDataTableCollapsed = false;

  collapsedSectionNames = new Set<string>();
}
