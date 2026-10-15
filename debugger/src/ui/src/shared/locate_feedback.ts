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

/** Shared feedback for locating a token or chart region without changing selection. */
export function animateLocationFeedback(target: HTMLElement, reducedMotion: boolean): Animation {
  const highlight = 'inset 0 0 0 2px var(--me-primary-color)';
  return target.animate(
    reducedMotion
      ? [{boxShadow: highlight}, {boxShadow: highlight}]
      : Array.from({length: 11}, (_, i) => ({
          transform: `translateX(${i === 0 || i === 10 ? 0 : (i % 2 ? 1 : -1) * 4 * (1 - i / 11)}px)`,
          boxShadow: i === 10 ? 'inset 0 0 0 2px transparent' : highlight,
        })),
    {duration: reducedMotion ? 1800 : 1100, delay: 250, easing: 'ease-in-out', fill: 'backwards'},
  );
}

/** Analysis locators emphasize the destination without moving chart content or controls. */
export function animateAnalysisLocationFeedback(
  target: HTMLElement,
  reducedMotion: boolean,
): Animation {
  const highlight = 'inset 0 0 0 2px var(--me-primary-color)';
  return target.animate(
    reducedMotion
      ? [{boxShadow: highlight}, {boxShadow: highlight}]
      : [
          {boxShadow: highlight, offset: 0},
          {boxShadow: highlight, offset: 0.8},
          {boxShadow: 'inset 0 0 0 2px transparent', offset: 1},
        ],
    {duration: 1800, delay: 250, easing: 'ease-out', fill: 'backwards'},
  );
}
