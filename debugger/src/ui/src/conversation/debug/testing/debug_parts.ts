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

import {ConversationState} from '../../conversation_state';
import {ConversationViews} from '../saved_views/conversation_views';
import {DetailsInfo} from '../details_panel/details_info';
import {AnalysisInfo} from '../stage_analysis_rail/analysis_info';
import {AnalysisPanels} from '../stage_analysis_rail/analysis_panels';
import {StageAnalyses} from '../stage_analysis_rail/stage_analyses';
import {TokenDiff} from '../token_diff/token_diff';
import {TokenDiffDom} from '../token_diff/token_diff_dom';

/** Fields to give each part, by the names its class uses. */
export interface PartFields {
  conversation?: object;
  tokens?: object;
  panels?: object;
  analyses?: object;
  analysisInfo?: object;
  detailsInfo?: object;
  dom?: object;
  views?: object;
}

/**
 * The Debug reading's parts over stubbed state, without Angular: each part is an object of its
 * class with only the fields given, and the parts it uses, so a test runs the real methods of
 * the parts it calls on the state it sets.
 */
export function debugParts(fields: PartFields = {}) {
  const part = <T>(type: abstract new (...args: never[]) => T, ...values: (object | undefined)[]) =>
    Object.assign(Object.create(type.prototype), ...values) as T;
  const conversation = part(ConversationState, fields.conversation);
  const tokens = part(TokenDiff, {conversation}, fields.tokens);
  const panels = part(AnalysisPanels, {conversation, tokens}, fields.panels);
  const analyses = part(StageAnalyses, {conversation, tokens, panels}, fields.analyses);
  const analysisInfo = part(
    AnalysisInfo,
    {conversation, tokens, panels, analyses},
    fields.analysisInfo,
  );
  const detailsInfo = part(DetailsInfo, {conversation, tokens}, fields.detailsInfo);
  const dom = part(TokenDiffDom, {conversation, tokens, panels}, fields.dom);
  const views = part(ConversationViews, {conversation, tokens, panels, analysisInfo}, fields.views);
  return {conversation, tokens, panels, analyses, analysisInfo, detailsInfo, dom, views};
}
