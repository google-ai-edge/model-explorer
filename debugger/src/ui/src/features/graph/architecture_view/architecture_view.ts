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
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  ViewEncapsulation,
  effect,
  inject,
  input,
  output,
  viewChild,
} from '@angular/core';
import type {ComparisonRow, Semantic} from '../../../data/contracts/types';
import {
  ArchitectureRenderer,
  type ArchitectureAction,
  type ArchitectureSelection,
  type ArchitectureViewport,
} from './architecture_renderer';

export type {
  ArchitectureAction,
  ArchitectureSelection,
} from './architecture_renderer';

/** The accepted vertical architecture canvas, backed by the selected report. */
@Component({
  selector: 'graph-architecture-view',
  standalone: true,
  template: '<div #canvas class="architectureView"></div>',
  styleUrl: './architecture_view.scss',
  encapsulation: ViewEncapsulation.None,
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ArchitectureView {
  readonly model = input.required<Semantic>();
  readonly layer = input(0);
  readonly rows = input<ComparisonRow[]>([]);
  readonly metric = input('CosSim');
  readonly selection = input<ArchitectureSelection | null>(null);
  readonly inspect = output<string>();
  readonly execute = output<string>();
  readonly restoreViewport = input<ArchitectureViewport | null>(null);
  readonly viewportChanged = output<ArchitectureViewport>();
  private readonly canvas = viewChild<ElementRef<HTMLDivElement>>('canvas');
  private renderer?: ArchitectureRenderer;
  private restoredViewport?: ArchitectureViewport;

  constructor() {
    effect(() => {
      const options = {
        model: this.model(),
        layer: this.layer(),
        rows: this.rows(),
        metric: this.metric(),
        selection: this.selection(),
      };
      const host = this.canvas()?.nativeElement;
      if (!host) return;
      this.renderer ??= new ArchitectureRenderer(host, {
        inspect: (id) => this.inspect.emit(id),
        execute: (id) => this.execute.emit(id),
        viewportChanged: (value) => this.viewportChanged.emit(value),
      });
      this.renderer.render(options);
      const viewport = this.restoreViewport();
      if (viewport && viewport !== this.restoredViewport) {
        this.renderer.restoreViewport(viewport);
        this.restoredViewport = viewport;
      }
    });
    inject(DestroyRef).onDestroy(() => this.renderer?.destroy());
  }

  /** Explicit navigation only: changing an input never fits or recenters the graph. */
  control(action: ArchitectureAction, focusId?: string): boolean {
    return this.renderer?.control(action, focusId) ?? false;
  }
}
