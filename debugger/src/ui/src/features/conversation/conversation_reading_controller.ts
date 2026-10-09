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
  captureReadingPosition,
  readingPositionKey,
  savedReadingPosition,
  switchReadingPosition,
  type ConversationMode,
  type ConversationObservation,
  type ConversationReadingPosition,
} from './conversation_view_state';
import type {FlowVirtualizer} from './flow_virtualization';

type ReadingSnapshot = ConversationReadingPosition & {mode: ConversationMode};
interface ReadingOptions {
  context: string;
  positions: Map<string, ConversationReadingPosition>;
  root: () => HTMLElement;
  mode: () => ConversationMode;
  virtualizer: () => FlowVirtualizer | null;
  /** Schedule after Angular renders; the controller rejects superseded work. */
  afterRender: (work: () => void) => void;
  afterRestore: () => void;
}

/** Owns mode-specific reading positions and rendered-layout restoration. */
export class ConversationReadingController {
  private activeMode: ConversationMode | null = null;
  private pendingMode: ConversationMode | null = null;
  private revision = 0;
  private disposed = false;
  private observation: ConversationObservation | null = null;
  private lastCapture: ReadingSnapshot | null = null;

  constructor(private readonly options: ReadingOptions) {}

  capture(): ReadingSnapshot {
    if (this.disposed && this.lastCapture)
      return structuredClone(this.lastCapture);
    return {
      mode: this.activeMode ?? this.options.mode(),
      ...captureReadingPosition(
        this.options.root().querySelector('.reading')?.scrollTop ?? 0,
        this.options.virtualizer()?.capture(false) ?? null,
      ),
    };
  }

  apply(
    snapshot: ReadingSnapshot,
    replace: boolean,
    observation: ConversationObservation | null,
  ): void {
    if (this.disposed) return;
    const key = readingPositionKey(this.options.context, snapshot.mode);
    if (replace || !this.options.positions.has(key)) {
      this.options.positions.set(
        key,
        captureReadingPosition(snapshot.scrollTop, snapshot.anchor),
      );
    }
    this.observation = observation;
    this.restore(
      savedReadingPosition(
        this.options.positions,
        this.options.context,
        this.options.mode(),
      ),
    );
  }

  syncMode(): void {
    if (this.disposed) return;
    const mode = this.options.mode();
    if (this.activeMode === mode && this.pendingMode === null) return;
    const incoming = switchReadingPosition(
      this.options.positions,
      this.options.context,
      this.activeMode,
      mode,
      this.capture(),
    );
    this.restore(incoming);
  }

  /** Includes pinned headings, composer, and overlay inspector in the readable area. */
  visibleBounds(): {top: number; bottom: number} | null {
    const root = this.options.root(),
      reading = root.querySelector('.reading');
    if (!reading) return null;
    const viewport = reading.getBoundingClientRect();
    const heading = reading
      .querySelector('.column-headings')
      ?.getBoundingClientRect();
    const title =
      this.options.mode() === 'Debug'
        ? reading.querySelector('.nodeTitle')
        : null;
    const titleHeight = title
      ? (heading?.height ?? 60) +
        (parseFloat(
          getComputedStyle(root).getPropertyValue('--node-pinned-height'),
        ) || 40)
      : 0;
    const top = viewport.top + Math.max(heading?.height ?? 0, titleHeight) + 8;
    let bottom =
      Math.min(
        viewport.bottom,
        root.querySelector('.composer')?.getBoundingClientRect().top ??
          viewport.bottom,
      ) - 16;
    const inspector = root
      .querySelector('.token-inspector')
      ?.getBoundingClientRect();
    if (
      inspector &&
      inspector.left < viewport.right &&
      inspector.right > viewport.left &&
      inspector.top > top
    )
      bottom = Math.min(bottom, inspector.top - 8);
    return {top, bottom: Math.max(top + 26, bottom)};
  }

  private restore(position: ConversationReadingPosition): void {
    const revision = ++this.revision,
      mode = this.options.mode(),
      observation = this.observation;
    this.pendingMode = mode;
    this.options.afterRender(() => {
      if (
        this.disposed ||
        revision !== this.revision ||
        mode !== this.options.mode()
      )
        return;
      this.activeMode = mode;
      this.pendingMode = null;
      const root = this.options.root(),
        reading = root.querySelector('.reading');
      if (!reading) return;
      const virtualizer = this.options.virtualizer();
      virtualizer?.mount();
      reading.scrollTop = position.scrollTop;
      if (!observation && position.anchor)
        virtualizer?.restore(position.anchor);
      else virtualizer?.paint(true);
      if (observation) this.locateObservation(reading, observation);
      this.observation = null;
      this.options.afterRestore();
      const draft = root.querySelector('textarea');
      if (draft) {
        draft.style.height = 'auto';
        draft.style.height = Math.min(120, draft.scrollHeight) + 'px';
      }
      // Keep the new mode's position after any observation-based navigation.
      const captured = this.capture();
      this.options.positions.set(
        readingPositionKey(this.options.context, mode),
        captureReadingPosition(captured.scrollTop, captured.anchor),
      );
    });
  }

  private locateObservation(
    reading: Element,
    observation: ConversationObservation,
  ): void {
    if (!Number.isInteger(observation.turn) || observation.turn < 1) return;
    const turn = observation.turn;
    const phase = ['prefill', 'decode'].includes(observation.phase)
      ? observation.phase
      : null;
    const node =
      (this.options.mode() === 'Debug' && phase
        ? reading.querySelector(
            `[data-node-turn="${turn}"][data-node-stage="${phase}"]`,
          )
        : null) ?? reading.querySelector(`.turn[data-turn="${turn}"]`);
    if (!node) return;
    const top =
      this.visibleBounds()?.top ?? reading.getBoundingClientRect().top;
    reading.scrollTop += node.getBoundingClientRect().top - top;
    this.options.virtualizer()?.paint(true);
  }

  /** Capture before invalidating layout, and cancel any pending restore callback. */
  destroy(): void {
    if (this.disposed) return;
    this.lastCapture = this.capture();
    this.options.positions.set(
      readingPositionKey(this.options.context, this.lastCapture.mode),
      captureReadingPosition(
        this.lastCapture.scrollTop,
        this.lastCapture.anchor,
      ),
    );
    this.disposed = true;
    this.revision++;
    this.options.virtualizer()?.destroy();
  }
}
