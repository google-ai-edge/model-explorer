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

import {ChangeDetectionStrategy, Component, computed, inject, signal} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MAT_DIALOG_DATA, MatDialogModule} from '@angular/material/dialog';

import {ConversationState} from '../../../conversation_state';
import type {StageAnalysisPanel} from '../../../stage_analysis';
import {AnalysisArchitecture} from '../analysis_architecture/analysis_architecture';
import {AnalysisPanels} from '../analysis_panels';

/** The stage a picker adds an analysis view to. */
export interface AnalysisPickerData {
  turn: number;
  stage: 'prefill' | 'decode';
}

/** The "Add analysis view" dialog: the model's architecture, whose objects are the analyses a
 *  stage offers; it closes with the chosen kind. */
@Component({
  selector: 'analysis-picker',
  imports: [AnalysisArchitecture, MatButtonModule, MatDialogModule],
  templateUrl: './analysis_picker.ng.html',
  styleUrl: './analysis_picker.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AnalysisPicker {
  protected readonly data = inject<AnalysisPickerData>(MAT_DIALOG_DATA);
  protected readonly conversation = inject(ConversationState);
  protected readonly panels = inject(AnalysisPanels);
  readonly analysisOptions = [
    {
      kind: 'tokens',
      label: 'Final Logits',
      splitBy: '-',
      breakdownBy: '-',
      aggregation: '-',
      icon: 'show_chart',
    },
    {
      kind: 'distribution',
      label: 'Sampling Distribution',
      splitBy: '-',
      breakdownBy: 'Runtime',
      aggregation: '-',
      icon: 'show_chart',
    },
    {
      kind: 'activations',
      label: 'Intermediate Tensors',
      splitBy: '-',
      breakdownBy: 'Layer',
      aggregation: 'Sync points',
      icon: 'analytics',
    },
    {
      kind: 'kv',
      label: 'KV cache',
      splitBy: 'Cache type',
      breakdownBy: 'Layer',
      aggregation: 'Heads',
      icon: 'grid_on',
    },
  ] as const;
  readonly analysisPickerKind = signal<StageAnalysisPanel['kind'] | null>(null);
  readonly analysisPickerOption = computed(() =>
    this.analysisOptions.find((option) => option.kind === this.analysisPickerKind()),
  );
  analysisArchitectureUnavailable(
    turn: number,
    stage: string,
  ): Record<StageAnalysisPanel['kind'], string> {
    return {
      tokens: this.panels.analysisUnavailable(turn, stage, 'tokens'),
      distribution: this.panels.analysisUnavailable(turn, stage, 'distribution'),
      activations: this.panels.analysisUnavailable(turn, stage, 'activations'),
      kv: this.panels.analysisUnavailable(turn, stage, 'kv'),
    };
  }
}
