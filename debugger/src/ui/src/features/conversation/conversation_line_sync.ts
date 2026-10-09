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

/** Shared line breaks for the static Debug rows.
 *
 * Both runtimes render the same aligned pairs, but a gap (∅) or a differing token is not
 * as wide as its partner, so the two columns would wrap at different tokens. Giving every
 * pair the width of its wider side keeps corresponding tokens on the same line, as the
 * virtualized rows already do through their shared packing.
 */
const SELECTOR =
  '.tokens button[data-row][data-turn][data-side], .input-tokens [data-row][data-turn][data-side]';

export function syncPairWidths(root: ParentNode): number {
  const elements = Array.from(root.querySelectorAll<HTMLElement>(SELECTOR));
  if (!elements.length) return 0;
  for (const element of elements) {
    element.style.minWidth = '';
    element.style.marginBottom = '';
  }
  // One layout pass reads every natural box before any size is written back.
  const boxes = new Map(
    elements.map((element) => [element, element.getBoundingClientRect()]),
  );
  const groups = new Map<string, HTMLElement[]>();
  for (const element of elements) {
    const key = `${element.dataset['kind'] ?? 'output'}:${element.dataset['turn']}:${element.dataset['row']}`;
    (groups.get(key) ?? groups.set(key, []).get(key)!).push(element);
  }
  let adjusted = 0;
  for (const group of groups.values()) {
    if (group.length < 2) continue;
    const widest = Math.max(
      ...group.map((element) => boxes.get(element)!.width),
    );
    // A token that wraps inside its own box is taller than its partner; equal line heights keep
    // the following tokens of both runtimes on the same line. The shorter one grows by a bottom
    // margin, not by its height: a taller box would recentre its text below the line.
    const tallest = Math.max(
      ...group.map((element) => boxes.get(element)!.height),
    );
    for (const element of group) {
      const box = boxes.get(element)!;
      if (widest - box.width > 0.5) {
        element.style.minWidth = `${widest}px`;
        adjusted++;
      }
      if (tallest - box.height > 0.5) {
        element.style.marginBottom = `${tallest - box.height}px`;
        adjusted++;
      }
    }
  }
  return adjusted;
}
