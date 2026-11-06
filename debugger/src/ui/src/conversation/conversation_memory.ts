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
import type {ConversationReadingPosition, TokenDiffViewState} from './conversation_view_state';

/**
 * What each chat's conversation was left with, by its capture (ConversationState.contextKey),
 * kept while the application runs so that coming back to a chat returns to it: its Token Diff
 * view and where each reading was.
 */
@Injectable({providedIn: 'root'})
export class ConversationMemory {
  readonly conversationViews = new Map<string, TokenDiffViewState>();
  readonly readingPositions = new Map<string, ConversationReadingPosition>();
}
