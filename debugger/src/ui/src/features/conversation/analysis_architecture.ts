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

import {ChangeDetectionStrategy, Component, computed, input, output} from '@angular/core';
import {NgTemplateOutlet} from '@angular/common';
import type {StageAnalysisPanel} from './stage_analysis';

type AnalysisKind = StageAnalysisPanel['kind'];
interface AnalysisObject {
  kind: AnalysisKind;
  label: string;
  x: number;
  y: number;
  module: string;
  insideModule?: boolean;
}

/** Semantic orientation only; this is not an operator graph or tensor mapping. */
@Component({
  selector: 'analysis-architecture',
  standalone: true,
  imports: [NgTemplateOutlet],
  templateUrl: './analysis_architecture.ng.html',
  styleUrl: './analysis_architecture.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AnalysisArchitecture {
  readonly model = input.required<string>();
  // Capture annotations describe instrumentation, not the model's name.
  readonly modelName = computed(
    () =>
      this.model()
        .replace(/\s+\(tapped(?:,\s*[^()]*)?\)\s*$/i, '')
        .trim() || 'Model',
  );
  readonly stage = input.required<'prefill' | 'decode'>();
  readonly selected = input<AnalysisKind | null>(null);
  readonly unavailable = input<Partial<Record<AnalysisKind, string>>>({});
  readonly selectionChange = output<AnalysisKind | null>();
  readonly objects = computed<AnalysisObject[]>(() => [
    {kind: 'kv', label: 'KV cache', x: 260, y: 114, module: 'layers'},
    {
      kind: 'activations',
      label: 'Intermediate Tensors',
      x: 10,
      y: 36,
      module: 'layers',
      insideModule: true,
    },
    ...(this.stage() === 'decode'
      ? [
          {kind: 'tokens' as const, label: 'Logits', x: 260, y: 238, module: 'head'},
          {
            kind: 'distribution' as const,
            label: 'Sampling distribution',
            x: 260,
            y: 286,
            module: 'sampling',
          },
        ]
      : []),
  ]);
  readonly selectedObject = computed(() =>
    this.objects().find((item) => item.kind === this.selected()),
  );
  readonly modules = computed(() => [
    {id: 'embedding', label: 'Token embedding', x: 64, y: 56, width: 142, height: 32},
    {id: 'layers', label: 'Transformer layers', x: 40, y: 112, width: 190, height: 111},
    ...(this.stage() === 'decode'
      ? [
          {id: 'head', label: 'LM head', x: 87, y: 238, width: 96, height: 32},
          {id: 'sampling', label: 'Sampling', x: 85, y: 286, width: 100, height: 32},
        ]
      : []),
  ]);
  readonly spine = computed(() =>
    this.stage() === 'decode'
      ? ['M135 38V56', 'M135 88V112', 'M135 223V238', 'M135 270V286', 'M135 318V334']
      : ['M135 38V56', 'M135 88V112'],
  );
  readonly branches = computed(() => [
    {kind: 'kv', d: 'M238 130H260', x: 260, y: 130},
    ...(this.stage() === 'decode'
      ? [
          {kind: 'tokens', d: 'M183 254H260', x: 260, y: 254},
          {kind: 'distribution', d: 'M185 302H260', x: 260, y: 302},
        ]
      : []),
  ]);
  private blankPress: {x: number; y: number} | null = null;
  startBlank(event: PointerEvent) {
    this.blankPress =
      event.button === 0 && !(event.target as Element).closest('[data-node]')
        ? {x: event.clientX, y: event.clientY}
        : null;
  }
  endBlank(event: PointerEvent) {
    if (
      this.blankPress &&
      Math.hypot(event.clientX - this.blankPress.x, event.clientY - this.blankPress.y) < 5 &&
      !(event.target as Element).closest('[data-node]')
    )
      this.selectionChange.emit(null);
    this.blankPress = null;
  }
  cancelBlank() {
    this.blankPress = null;
  }
}
