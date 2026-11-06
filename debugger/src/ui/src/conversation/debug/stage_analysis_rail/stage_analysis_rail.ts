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
  inject,
  input,
  viewChildren,
} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatTooltipModule} from '@angular/material/tooltip';

import {ANALYSES, GraphMetricPanel, KvPanel, analysisOf} from '../../../analyses/registry';
import {ConversationState} from '../../conversation_state';
import {TokenDiff} from '../token_diff/token_diff';
import {AnalysisInfo} from './analysis_info';
import {AnalysisPanels, type AnalysisRail} from './analysis_panels';
import {AnalysisTokenStrip} from './analysis_token_strip/analysis_token_strip';
import {ExpandedAnalysis} from './expanded_analysis';
import {LogitPanel} from './logit_panel';
import {StageAnalyses} from './stage_analyses';

/**
 * One stage's rail of analysis views beside its Token Diff nodes: each view's entry, which opens,
 * closes or removes it, its panel, and the button that adds another; or, for the expanded stage,
 * its views as tabs over the page with the stage's token strip. A view whose first data is still
 * loading cannot open, and shows a spinner where its remove button goes.
 */
@Component({
  selector: 'stage-analysis-rail',
  imports: [
    ANALYSES.intermediate_tensors.panel,
    ANALYSES.kv.panel,
    AnalysisTokenStrip,
    LogitPanel,
    MatIconModule,
    MatProgressSpinnerModule,
    MatTooltipModule,
  ],
  templateUrl: './stage_analysis_rail.ng.html',
  styleUrl: './stage_analysis_rail.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StageAnalysisRail implements AnalysisRail {
  readonly turn = input.required<number>();
  readonly stage = input.required<'prefill' | 'decode'>();
  protected readonly conversation = inject(ConversationState);
  protected readonly tokens = inject(TokenDiff);
  protected readonly panels = inject(AnalysisPanels);
  protected readonly analyses = inject(StageAnalyses);
  protected readonly expanded = inject(ExpandedAnalysis);
  protected readonly analysisInfo = inject(AnalysisInfo);
  protected readonly analysisOf = analysisOf;
  readonly kvPanels = viewChildren(KvPanel);
  readonly graphPanels = viewChildren(GraphMetricPanel);
  readonly tokenPanels = viewChildren(LogitPanel);
  constructor() {
    inject(DestroyRef).onDestroy(this.panels.addRail(this));
  }
}
