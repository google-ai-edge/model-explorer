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

import {ChangeDetectionStrategy, Component, computed, inject, input} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatSlideToggleModule} from '@angular/material/slide-toggle';
import {MatTooltipModule} from '@angular/material/tooltip';
import {WorkspaceStateService} from '../../../state/workspace_state_service';
import {ConversationComposerController} from '../conversation_composer';

/**
 * The message composer under the conversation: in Chat the message box and Send (or Stop while a
 * reply streams), in both modes the Debug switch, which goes only to a mode the server allows. The
 * page puts the Debug reading's tools in it: a row above, `[composer-tools]`, and the bottom row's
 * status, `[composer-status]`.
 */
@Component({
  selector: 'message-composer',
  imports: [MatButtonModule, MatIconModule, MatSlideToggleModule, MatTooltipModule],
  templateUrl: './message_composer.ng.html',
  styleUrl: './message_composer.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class MessageComposer {
  /**
   * The width the details panel takes beside the conversation; the Debug composer centers
   * itself in the rest.
   */
  readonly inspectorWidth = input(0);
  protected readonly workspace = inject(WorkspaceStateService);
  protected readonly composer = inject(ConversationComposerController);
  /** Whether the server allows the mode the Debug switch goes to, and if not why. */
  protected readonly switchTo = computed(
    () => this.workspace.modes()[this.workspace.mode() === 'Debug' ? 'Chat' : 'Debug'],
  );
}
