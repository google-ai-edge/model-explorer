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

import {
  animate,
  AnimationTriggerMetadata,
  state,
  style,
  transition,
  trigger,
} from '@angular/animations';

/** Transition time of bubble animations. */
export const ANIMATION_TRANSITION_TIME_MS = 200;

const HIDDEN_STYLE = style({opacity: 0});
const VISIBLE_STYLE = style({opacity: 1});

const ENTERING_TIMING = `${ANIMATION_TRANSITION_TIME_MS}ms cubic-bezier(0.0,0.0,0.2,1)`;
const LEAVING_TIMING = `${ANIMATION_TRANSITION_TIME_MS}ms cubic-bezier(0.4,0.0,0.2,1)`;

/** Animations used by bubble. */
export const BUBBLE_ANIMATIONS: {
  readonly bubbleContainer: AnimationTriggerMetadata;
} = {
  /** Animation that is applied on the bubble container by default. */
  bubbleContainer: trigger('bubbleContainer', [
    state('void, hidden', HIDDEN_STYLE),
    state('visible', VISIBLE_STYLE),
    transition(
      'void => *, * => visible',
      animate(ENTERING_TIMING, VISIBLE_STYLE),
    ),
    transition('* => void, * => hidden', animate(LEAVING_TIMING, HIDDEN_STYLE)),
  ]),
};
