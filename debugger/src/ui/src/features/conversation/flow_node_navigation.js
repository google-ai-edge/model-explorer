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
export function createFlowNodeNavigator(config) {
  let scroller = null, raf = 0, resizeObserver = null, pairs = [],
      destroyed = false;
  const PIN_INSET = 0;
  const PIN_HEIGHT = 40;
  const PUSH_GAP = 0;

  function pairBounds(pair) {
    const viewport = scroller.getBoundingClientRect(), rects = [
      pair.ref, pair.target
    ].filter(Boolean).map((node) => node.getBoundingClientRect());
    const top = Math.min(...rects.map((rect) => rect.top)),
          bottom = Math.max(...rects.map((rect) => rect.bottom));
    return {top, bottom, height: bottom - top, viewport};
  }

  function clearPinned(keep) {
    config.getRoot()
        .querySelectorAll('.nodeTitle.isPinned')
        .forEach((title) => {
          if (keep &&
              (title.closest('[data-node-stage]') === keep.ref ||
               title.closest('[data-node-stage]') === keep.target))
            return;
          title.closest('[data-node-stage]').classList.remove('nodePinned');
          title.closest('[data-node-stage]')
              .style.removeProperty('--node-visible-top');
          title.classList.remove('isPinned');
          title.style.removeProperty('top');
          title.style.removeProperty('left');
          title.style.removeProperty('width');
          title.style.removeProperty('--pinned-node-surface');
        });
  }

  function collect() {
    const refs = scroller ?
        [
          ...scroller.querySelectorAll(
              '.conversation-side[data-runtime="ref"] :is(.inputNode,.decodeNode)',
              ),
        ] :
        [];
    pairs = refs.map((ref) => {
      const turn = Number(ref.dataset.nodeTurn), stage = ref.dataset.nodeStage;
      return {
        turn,
        stage,
        ref,
        target: scroller.querySelector(
            `.conversation-side[data-runtime="target"] [data-node-turn="${
                turn}"][data-node-stage="${stage}"]`,
            ),
      };
    });
  }

  function activeIndex() {
    if (!pairs.length) return -1;
    const viewport = scroller.getBoundingClientRect(),
          probe = viewport.top + PIN_INSET + 1;
    let best = -1, bestOverlap = -1;
    pairs.forEach((pair, index) => {
      const box = pairBounds(pair);
      if (box.top <= probe && box.bottom >= probe + PIN_HEIGHT) {
        best = index;
        bestOverlap = Infinity;
        return;
      }
      if (bestOverlap === Infinity) return;
      const overlap = Math.max(
          0,
          Math.min(box.bottom, viewport.bottom) -
              Math.max(box.top, viewport.top),
      );
      if (overlap > bestOverlap) {
        best = index;
        bestOverlap = overlap;
      }
    });
    return best;
  }

  function preparePinned(title, node, top) {
    const style = getComputedStyle(node),
          transparent =
              /^(transparent|rgba\([^)]*,\s*0\))$/.test(style.backgroundColor);
    const side = node.dataset.nodeSide,
          rootStyle = getComputedStyle(config.getRoot()),
          outside = rootStyle
                        .getPropertyValue(
                            side === 'ref' ? '--reference-surface' :
                                             '--target-debug-surface')
                        .trim(),
          surface = transparent ? outside : style.backgroundColor;
    return () => {
      node.classList.add('nodePinned');
      // Native sticky positioning follows scrolling without a per-frame offset
      // write.
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
    if (!pairs.length || !pairs[0].ref.isConnected) collect();
    const index = activeIndex();
    if (index < 0) {
      clearPinned();
      return;
    }
    const pair = pairs[index], box = pairBounds(pair),
          longNode = box.height >= scroller.clientHeight - 32;
    config.getRoot().querySelectorAll('.nodeJumpControls').forEach((group) => {
      const hide = group.closest('[data-node-stage]') !== pair.target ||
          !longNode || config.isCollapsed(pair.turn, pair.stage);
      if (group.hidden !== hide) group.hidden = hide;
    });
    if (!longNode || config.isCollapsed(pair.turn, pair.stage)) {
      clearPinned();
      return;
    }
    const runtimeBottom = config.getRoot()
                              .querySelector('.column-headings')
                              ?.getBoundingClientRect()
                              .bottom ||
        box.viewport.top;
    const stickyTop = Math.max(box.viewport.top, runtimeBottom) + PIN_INSET;
    if (box.top > stickyTop || box.bottom <= stickyTop) {
      clearPinned();
      return;
    }
    clearPinned(pair);
    const top = Math.min(stickyTop, box.bottom - PIN_HEIGHT - PUSH_GAP);
    const writes = [];
    for (const side of ['ref', 'target']) {
      const node = pair[side], title = node?.querySelector('.nodeTitle');
      if (node && title) writes.push(preparePinned(title, node, top));
    }
    writes.forEach((write) => write());
  }

  function schedule() {
    if (!destroyed && !raf) raf = requestAnimationFrame(update);
  }

  function jump(index, action) {
    const pair = pairs[index];
    if (!pair) return;
    let target = pair;
    if (action === 'previous') target = pairs[index - 1];
    if (action === 'next') target = pairs[index + 1];
    if (!target) return;
    const box = pairBounds(target), inset = 12,
          node = target.target || target.ref;
    const before = node.previousElementSibling, after = node.nextElementSibling;
    const start = before?.matches('.stageActions,.stageActionsSpacer') ?
        Math.min(box.top, before.getBoundingClientRect().top) :
        box.top;
    const end = after?.matches('.stageActions,.stageActionsSpacer') ?
        Math.max(box.bottom, after.getBoundingClientRect().bottom) :
        box.bottom;
    const dock =
        config.getRoot().querySelector('.composer')?.getBoundingClientRect();
    const viewportBottom =
        dock && dock.height > 0 && dock.top > box.viewport.top ?
        Math.min(box.viewport.bottom, dock.top) :
        box.viewport.bottom;
    scroller.scrollTop = action === 'end' ?
        Math.max(0, scroller.scrollTop + end - viewportBottom + inset) :
        Math.max(0, scroller.scrollTop + start - box.viewport.top - inset);
    config.afterJump?.();
    schedule();
  }

  function handleClick(event) {
    const jumpControl = event.target.closest('[data-node-jump]');
    if (jumpControl && !jumpControl.disabled) {
      event.stopPropagation();
      jump(Number(jumpControl.dataset.nodeIndex), jumpControl.dataset.nodeJump);
      return;
    }
    const toggle = event.target.closest('[data-node-toggle]');
    if (toggle) {
      event.stopPropagation();
      const pair = pairs[Number(toggle.dataset.nodeIndex)];
      if (pair) config.toggleCollapsed(pair.turn, pair.stage);
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
    resizeObserver = new ResizeObserver(schedule);
    if (scroller) resizeObserver.observe(scroller);
    pairs.flatMap((pair) => [pair.ref, pair.target])
        .filter(Boolean)
        .forEach((node) => resizeObserver.observe(node));
    requestAnimationFrame(() => requestAnimationFrame(schedule));
  }

  return {
    destroy() {
      destroyed = true;
      cancelAnimationFrame(raf);
      resizeObserver?.disconnect();
      scroller?.removeEventListener('scroll', schedule);
      scroller?.removeEventListener('click', handleClick);
      clearPinned();
    },
    mount,
    update: schedule,
    jumpStage: (turn, stage, action = 'start') => {
      const index =
          pairs.findIndex((pair) => pair.turn === turn && pair.stage === stage);
      if (index >= 0) jump(index, action);
    },
  };
}
