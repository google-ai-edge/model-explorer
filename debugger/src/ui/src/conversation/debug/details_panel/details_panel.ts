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

import {NgTemplateOutlet} from '@angular/common';
import {ChangeDetectionStrategy, Component, inject} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import {ExplorerInfoList} from '../../../widgets/explorer_info_list/explorer_info_list';
import {ExplorerInfoSection} from '../../../widgets/explorer_info_section/explorer_info_section';
import {ExplorerInfoValue} from '../../../widgets/explorer_info_value/explorer_info_value';
import {ExplorerLocator} from '../../../widgets/explorer_locator/explorer_locator';
import {ExplorerPagination} from '../../../widgets/explorer_pagination/explorer_pagination';
import {ResizeHandle} from './resize_handle/resize_handle';
import {ConversationState} from '../../conversation_state';
import {StageStateInfo} from './stage_state_info/stage_state_info';
import type {TokenPair} from '../../token_alignment';
import {AnalysisInfo} from '../stage_analysis_rail/analysis_info';
import {AnalysisPanels} from '../stage_analysis_rail/analysis_panels';
import {StageAnalyses} from '../stage_analysis_rail/stage_analyses';
import {TokenDiff} from '../token_diff/token_diff';
import {TokenDiffDom} from '../token_diff/token_diff_dom';
import {DetailsInfo} from './details_info';

/**
 * The details panel beside the Debug reading, with its toggle and resize handle: the frame of the
 * analysis view that has the focus, or Conversation Info and the selected tokens' info with the
 * selected stage's analysis views.
 */
@Component({
  selector: 'details-panel',
  imports: [
    ExplorerInfoList,
    ExplorerInfoSection,
    ExplorerInfoValue,
    ExplorerLocator,
    ExplorerPagination,
    MatButtonModule,
    MatIconModule,
    MatTooltipModule,
    NgTemplateOutlet,
    ResizeHandle,
    StageStateInfo,
  ],
  templateUrl: './details_panel.ng.html',
  styleUrl: './details_panel.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DetailsPanel {
  protected readonly conversation = inject(ConversationState);
  protected readonly tokens = inject(TokenDiff);
  protected readonly panels = inject(AnalysisPanels);
  protected readonly analyses = inject(StageAnalyses);
  protected readonly analysisInfo = inject(AnalysisInfo);
  protected readonly detailsInfo = inject(DetailsInfo);
  protected readonly dom = inject(TokenDiffDom);

  activateTokenLocator(event: KeyboardEvent, pair: TokenPair): void {
    if (event.target !== event.currentTarget || !['Enter', ' '].includes(event.key)) {
      return;
    }
    event.preventDefault();
    this.locateInfoToken(pair);
  }
  locateInfoToken(pair: TokenPair): void {
    const selected = this.detailsInfo.selectionInfo();
    if (!selected) {
      return;
    }
    this.dom.locateSelectedToken({
      kind: selected.stage === 'prefill' ? 'input' : 'token',
      turn: selected.turn,
      index: pair.index,
    });
  }
}
