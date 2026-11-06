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

// Inherited from candidates flow-node-navigation.js. Angular owns title markup.

export interface FlowNodeNavigator {
  mount(): void;
  update(): void;
  jumpStage(turn: number, stage: string, action?: string): void;
  destroy(): void;
}
export interface FlowNodeNavigatorConfig {
  getRoot: () => HTMLElement;
  getScroller: () => HTMLElement | null;
  isEnabled: () => boolean;
  isCollapsed: (turn: number, stage: string) => boolean;
  toggleCollapsed: (turn: number, stage: string) => void;
  afterJump?: () => void;
}
/** One stage's nodes on both runtimes. */
interface NodePair {
  turn: number;
  stage: string;
  ref: HTMLElement;
  target: HTMLElement | null;
}

export function createFlowNodeNavigator(config: FlowNodeNavigatorConfig): FlowNodeNavigator {
  let scroller: HTMLElement | null = null;
  let raf = 0;
  let mountRaf = 0;
  let resizeObserver: ResizeObserver | null = null;
  let pairs: NodePair[] = [];
  let destroyed = false;
  const PIN_INSET = 0;
  const PIN_HEIGHT = 40;

  function pairBounds(pair: NodePair) {
    const viewport = scroller ? scroller.getBoundingClientRect() : new DOMRect();
    const rects = [pair.ref, pair.target]
      .filter((node): node is HTMLElement => !!node)
      .map((node) => node.getBoundingClientRect());
    const top = Math.min(...rects.map((rect) => rect.top));
    const bottom = Math.max(...rects.map((rect) => rect.bottom));
    return {top, bottom, height: bottom - top, viewport};
  }

  function clearPinned(keep?: NodePair) {
    for (const title of Array.from(
      config.getRoot().querySelectorAll<HTMLElement>('.nodeTitle.isPinned'),
    )) {
      const stageNode = title.closest<HTMLElement>('[data-node-stage]');
      if (keep && (stageNode === keep.ref || stageNode === keep.target)) {
        continue;
      }
      stageNode?.classList.remove('nodePinned');
      stageNode?.style.removeProperty('--node-visible-top');
      title.classList.remove('isPinned');
      title.style.removeProperty('top');
      title.style.removeProperty('left');
      title.style.removeProperty('width');
      title.style.removeProperty('--pinned-node-surface');
    }
  }

  function collect() {
    const activeScroller = scroller;
    if (!activeScroller) {
      pairs = [];
      return;
    }
    const refs = Array.from(
      activeScroller.querySelectorAll<HTMLElement>(
        '.conversation-side[data-runtime="ref"] :is(.inputNode,.decodeNode)',
      ),
    );
    pairs = refs.map((ref) => {
      const turn = Number(ref.dataset['nodeTurn']);
      const stage = ref.dataset['nodeStage'] ?? '';
      return {
        turn,
        stage,
        ref,
        target: activeScroller.querySelector<HTMLElement>(
          `.conversation-side[data-runtime="target"] [data-node-turn="${turn}"][data-node-stage="${stage}"]`,
        ),
      };
    });
  }

  function activeIndex() {
    if (!pairs.length || !scroller) {
      return -1;
    }
    const viewport = scroller.getBoundingClientRect();
    const probe = viewport.top + PIN_INSET + 1;
    let best = -1;
    let bestOverlap = -1;
    for (const [index, pair] of pairs.entries()) {
      const box = pairBounds(pair);
      if (box.top <= probe && box.bottom >= probe + PIN_HEIGHT) {
        best = index;
        bestOverlap = Infinity;
        continue;
      }
      if (bestOverlap === Infinity) {
        continue;
      }
      const overlap = Math.max(
        0,
        Math.min(box.bottom, viewport.bottom) - Math.max(box.top, viewport.top),
      );
      if (overlap > bestOverlap) {
        best = index;
        bestOverlap = overlap;
      }
    }
    return best;
  }

  function preparePinned(title: HTMLElement, node: HTMLElement) {
    const style = getComputedStyle(node);
    const transparent = /^(transparent|rgba\([^)]*,\s*0\))$/.test(style.backgroundColor);
    const side = node.dataset['nodeSide'];
    const rootStyle = getComputedStyle(config.getRoot());
    const outside = rootStyle
      .getPropertyValue(side === 'ref' ? '--reference-surface' : '--target-debug-surface')
      .trim();
    const surface = transparent ? outside : style.backgroundColor;
    return () => {
      node.classList.add('nodePinned');
      // Native sticky positioning follows scrolling without a per-frame offset write.
      title.classList.add('isPinned');
      title.style.setProperty('--pinned-node-surface', surface);
      title.style.setProperty('--node-corner-surface', outside);
    };
  }

  function update() {
    raf = 0;
    if (!scroller || !config.isEnabled()) {
      clearPinned();
      return;
    }
    if (!pairs.length || !pairs[0].ref.isConnected) {
      collect();
    }
    const index = activeIndex();
    if (index < 0) {
      clearPinned();
      return;
    }
    const pair = pairs[index];
    const box = pairBounds(pair);
    const longNode = box.height >= scroller.clientHeight - 32;
    for (const group of Array.from(
      config.getRoot().querySelectorAll<HTMLElement>('.nodeJumpControls'),
    )) {
      const hide =
        group.closest('[data-node-stage]') !== pair.target ||
        !longNode ||
        config.isCollapsed(pair.turn, pair.stage);
      if (group.hidden !== hide) {
        group.hidden = hide;
      }
    }
    if (!longNode || config.isCollapsed(pair.turn, pair.stage)) {
      clearPinned();
      return;
    }
    const runtimeBottom =
      config.getRoot().querySelector('.column-headings')?.getBoundingClientRect().bottom ||
      box.viewport.top;
    const stickyTop = Math.max(box.viewport.top, runtimeBottom) + PIN_INSET;
    if (box.top > stickyTop || box.bottom <= stickyTop) {
      clearPinned();
      return;
    }
    clearPinned(pair);
    const writes: (() => void)[] = [];
    for (const side of ['ref', 'target'] as const) {
      const node = pair[side];
      const title = node?.querySelector<HTMLElement>('.nodeTitle');
      if (node && title) {
        writes.push(preparePinned(title, node));
      }
    }
    for (const write of writes) {
      write();
    }
  }

  function schedule() {
    if (!destroyed && !raf) {
      raf = requestAnimationFrame(update);
    }
  }

  function jump(index: number, action: string) {
    const pair = pairs[index];
    if (!pair || !scroller) {
      return;
    }
    let target: NodePair | undefined = pair;
    if (action === 'previous') {
      target = pairs[index - 1];
    }
    if (action === 'next') {
      target = pairs[index + 1];
    }
    if (!target) {
      return;
    }
    const box = pairBounds(target);
    const inset = 12;
    const node = target.target || target.ref;
    const before = node.previousElementSibling;
    const after = node.nextElementSibling;
    const start = before?.matches('.stageActions,.stageActionsSpacer')
      ? Math.min(box.top, before.getBoundingClientRect().top)
      : box.top;
    const end = after?.matches('.stageActions,.stageActionsSpacer')
      ? Math.max(box.bottom, after.getBoundingClientRect().bottom)
      : box.bottom;
    const dock = config.getRoot().querySelector('.composer')?.getBoundingClientRect();
    const viewportBottom =
      dock && dock.height > 0 && dock.top > box.viewport.top
        ? Math.min(box.viewport.bottom, dock.top)
        : box.viewport.bottom;
    scroller.scrollTop =
      action === 'end'
        ? Math.max(0, scroller.scrollTop + end - viewportBottom + inset)
        : Math.max(0, scroller.scrollTop + start - box.viewport.top - inset);
    config.afterJump?.();
    schedule();
  }

  function handleClick(event: Event) {
    const target = event.target as Element;
    const jumpControl = target.closest<HTMLButtonElement>('[data-node-jump]');
    if (jumpControl && !jumpControl.disabled) {
      event.stopPropagation();
      jump(Number(jumpControl.dataset['nodeIndex']), jumpControl.dataset['nodeJump'] ?? 'start');
      return;
    }
    const toggle = target.closest<HTMLElement>('[data-node-toggle]');
    if (toggle) {
      event.stopPropagation();
      const pair = pairs[Number(toggle.dataset['nodeIndex'])];
      if (pair) {
        config.toggleCollapsed(pair.turn, pair.stage);
      }
    }
  }

  function mount() {
    const next = config.getScroller();
    if (scroller !== next) {
      scroller?.removeEventListener('scroll', schedule);
      scroller?.removeEventListener('click', handleClick);
      scroller = next;
      scroller?.addEventListener('scroll', schedule, {passive: true});
      scroller?.addEventListener('click', handleClick);
    }
    collect();
    resizeObserver?.disconnect();
    const observer = new ResizeObserver(schedule);
    resizeObserver = observer;
    if (scroller) {
      observer.observe(scroller);
    }
    for (const node of pairs
      .flatMap((pair) => [pair.ref, pair.target])
      .filter((n): n is HTMLElement => !!n)) {
      observer.observe(node);
    }
    cancelAnimationFrame(mountRaf);
    mountRaf = requestAnimationFrame(() => {
      if (destroyed) {
        return;
      }
      mountRaf = requestAnimationFrame(() => {
        mountRaf = 0;
        schedule();
      });
    });
  }

  return {
    destroy() {
      destroyed = true;
      cancelAnimationFrame(mountRaf);
      cancelAnimationFrame(raf);
      resizeObserver?.disconnect();
      scroller?.removeEventListener('scroll', schedule);
      scroller?.removeEventListener('click', handleClick);
      clearPinned();
    },
    mount,
    update: schedule,
    jumpStage: (turn, stage, action = 'start') => {
      const index = pairs.findIndex((pair) => pair.turn === turn && pair.stage === stage);
      if (index >= 0) {
        jump(index, action);
      }
    },
  };
}
