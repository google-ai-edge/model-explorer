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

/** Keep each aligned pair on the same visual line in the static Debug flow. */
const SELECTOR =
  '.tokens button[data-row][data-turn][data-side], .input-tokens [data-row][data-turn][data-side]';

export function syncPairWidths(root: ParentNode): number {
  const elements = Array.from(root.querySelectorAll<HTMLElement>(SELECTOR));
  if (!elements.length) return 0;
  for (const element of elements) {
    element.style.width = '';
    element.style.minWidth = '';
    element.style.marginBottom = '';
  }
  const boxes = new Map(elements.map((element) => [element, element.getBoundingClientRect()]));
  const groups = new Map<string, HTMLElement[]>();
  for (const element of elements) {
    const key = `${element.dataset['kind'] ?? 'output'}:${element.dataset['turn']}:${element.dataset['row']}`;
    (groups.get(key) ?? groups.set(key, []).get(key)!).push(element);
  }
  let adjusted = 0;
  const paired = [...groups.values()].filter((group) => group.length > 1);
  for (const group of paired) {
    // Give BOTH sides the same pixel-rounded width. Subpixel differences otherwise accumulate
    // until one runtime wraps a token earlier, especially after changing outlines or font metrics.
    const width = Math.ceil(Math.max(...group.map((element) => boxes.get(element)!.width)));
    if (!width) continue; // Collapsed stages are measured again when opened.
    for (const element of group) {
      element.style.width = `${width}px`;
      adjusted++;
    }
  }
  // Width changes can reflow a long token. Read heights after applying widths, not from the
  // natural boxes above, and retain the normal bottom margin when compensating the shorter side.
  const heights = new Map(
    paired.flat().map(
      (element) =>
        [
          element,
          {
            height: element.getBoundingClientRect().height,
            margin: parseFloat(getComputedStyle(element).marginBottom) || 0,
          },
        ] as const,
    ),
  );
  for (const group of paired) {
    const tallest = Math.max(...group.map((element) => heights.get(element)!.height));
    for (const element of group) {
      const {height, margin} = heights.get(element)!;
      if (tallest > height) {
        element.style.marginBottom = `${margin + tallest - height}px`;
        adjusted++;
      }
    }
  }
  return adjusted;
}
