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

import {Injectable} from '@angular/core';
import {
  graphViewKey,
  parseGraphView,
  type GraphViewState,
} from './graph_view_state';

@Injectable({providedIn: 'root'})
export class GraphViewStore {
  private readonly views = new Map<string, GraphViewState>();
  read(capture: string, batch: number): GraphViewState | null {
    return parseGraphView(this.views.get(graphViewKey(capture, batch)));
  }
  save(capture: string, batch: number, value: GraphViewState): void {
    const view = parseGraphView(value);
    if (!view) return;
    const key = graphViewKey(capture, batch);
    this.views.delete(key);
    this.views.set(key, view);
    if (this.views.size > 64)
      this.views.delete(this.views.keys().next().value!);
  }
}
