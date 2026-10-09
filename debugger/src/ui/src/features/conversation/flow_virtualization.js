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

// Vendored from the live candidates demo, flow-virtualization.js.
// Source SHA256:
// c9dd7226acd4626fddd9e2d6f3363f6ae182ce34a0d9a3c6c8ba1385f398ee14. Angular
// adapter: ES module, scoped lifecycle, cached resize anchors. Demo trend
// sampling is not bundled here; this component only renders text.

const OVERSCAN = 0.15;
export function createFlowVirtualizer(config) {
  const lineHeight = () => config.lineHeight?.() || 26;
  const threshold = config.threshold || 2048;
  const layouts = new Map();
  const widthCache = new Map();
  const inputRowCache = new WeakMap();
  const rowVersions = new WeakMap();
  let nextRowVersion = 0;
  let raf = 0, observer = null, scroller = null, painting = false,
      lastWidth = 0, clickHandler = null, destroyed = false, lastAnchor = null;

  function turn(ti) {
    return config.getTurn(ti);
  }
  function outputRows(ti) {
    return config.getRows(ti);
  }
  function inputTokens(ti, side) {
    return turn(ti).inputTokens?.[side] || [];
  }
  function inputRows(ti) {
    const t = turn(ti), kind = config.getAlignment();
    if (t.inputAlignments?.[kind]) return t.inputAlignments[kind];
    if (inputRowCache.has(t)) return inputRowCache.get(t);
    const ref = inputTokens(ti, 'ref'), target = inputTokens(ti, 'target'),
          length = Math.max(ref.length, target.length);
    const rows =
        Array.from({length}, (_, index) => ({
                               index,
                               ref: index < ref.length ? index : null,
                               target: index < target.length ? index : null,
                             }));
    inputRowCache.set(t, rows);
    return rows;
  }
  function usesOutput(ti) {
    const t = turn(ti);
    return (
        Math.max(t.steps.ref.length, t.steps.target.length) > threshold &&
        !!t.alignments?.[config.getAlignment()]);
  }
  function usesInput(ti) {
    return Math.max(
               inputTokens(ti, 'ref').length,
               inputTokens(ti, 'target').length) > threshold;
  }
  function host(side, ti, stage, kind = 'output') {
    return `<span class="virtualTextHost" data-v-side="${side}" data-v-turn="${
        ti}" data-v-stage="${stage}" data-v-kind="${kind}"></span>`;
  }
  function rowIndices(ti, stage, kind) {
    const rows = kind === 'input' ? inputRows(ti) : outputRows(ti);
    if (kind === 'input') return rows.map((_, i) => i);
    return rows.reduce((indices, row, index) => {
      for (const side of ['ref', 'target']) {
        const step = row[side];
        if (step != null && config.phase(ti, side, step) === stage) {
          indices.push(index);
          break;
        }
      }
      return indices;
    }, []);
  }
  function wordFor(ti, kind, side, index) {
    return kind === 'input' ? inputTokens(ti, side)[index] :
                              turn(ti).steps[side][index];
  }
  function textWidth(text, displayKey) {
    const key = displayKey + '\u0000' + text;
    if (widthCache.has(key)) return widthCache.get(key);
    if (/^[\r\n\u200b-\u200d\ufeff]*$/.test(text)) {
      widthCache.set(key, 10);
      return 10;
    }
    const canvas = textWidth.canvas ||
        (textWidth.canvas = document.createElement('canvas'));
    const context = canvas.getContext('2d');
    if (!textWidth.font) {
      const probe = document.createElement('span');
      probe.className = 'virtualTextWidthProbe';
      probe.textContent = 'mmmmmmmm iiii looks';
      document.body.append(probe);
      const style = getComputedStyle(probe);
      textWidth.font = style.font;
      context.font = textWidth.font;
      textWidth.scale = probe.getBoundingClientRect().width /
          Math.max(1, context.measureText(probe.textContent).width);
      probe.remove();
    }
    context.font = textWidth.font;
    const value = String(text), spaces = (value.match(/ /g) || []).length,
          plain = value.replaceAll(' ', '');
    const width =
        Math.max(
            1,
            context.measureText(plain).width * (textWidth.scale || 1) +
                spaces * 9.1) +
        (config.widthAdjustment?.() || 0);
    widthCache.set(key, width);
    return width;
  }
  function pack(rows, width, offset = 0) {
    const lines = [offset], lineByOffset = new Uint32Array(rows.length),
          gapWidths = new Float32Array(rows.length);
    let refUsed = 0, targetUsed = 0;
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      if ((refUsed + row.ref > width || targetUsed + row.target > width) &&
          (refUsed > 0 || targetUsed > 0)) {
        lines.push(offset + i);
        refUsed = 0;
        targetUsed = 0;
      }
      lineByOffset[i] = lines.length - 1;
      gapWidths[i] = row.gapWidth;
      refUsed += row.ref;
      targetUsed += row.target;
      if (row.newline && i + 1 < rows.length) {
        lines.push(offset + i + 1);
        refUsed = 0;
        targetUsed = 0;
      }
    }
    lines.push(offset + rows.length);
    return {lines, lineByOffset, gapWidths};
  }
  function layout(hostElement) {
    const ti = +hostElement.dataset.vTurn, stage = hostElement.dataset.vStage,
          kind = hostElement.dataset.vKind;
    const allRows = kind === 'input' ? inputRows(ti) : outputRows(ti);
    if (!rowVersions.has(allRows)) rowVersions.set(allRows, ++nextRowVersion);
    const width = Math.max(1, Math.floor(hostElement.clientWidth)),
          displayKey = config.displayKey(), key = [
            ti,
            stage,
            kind,
            width,
            rowVersions.get(allRows),
            kind === 'input' ? '' : config.getAlignment(),
            kind === 'input' ? '' : config.isDebug(),
            displayKey,
          ].join(':');
    if (layouts.has(key)) {
      const cached = layouts.get(key);
      layouts.delete(key);
      layouts.set(key, cached);
      return cached;
    }
    const indices = rowIndices(ti, stage, kind);
    const measured = indices.map((rowIndex) => {
      const row = allRows[rowIndex],
            refWord = row.ref == null ||
              (kind !== 'input' && config.phase(ti, 'ref', row.ref) !== stage) ?
          null :
          wordFor(ti, kind, 'ref', row.ref),
            targetWord = row.target == null ||
              (kind !== 'input' &&
               config.phase(ti, 'target', row.target) !== stage) ?
          null :
          wordFor(ti, kind, 'target', row.target);
      const refNatural = refWord == null ?
          null :
          Math.min(width, textWidth(refWord, displayKey));
      const targetNatural = targetWord == null ?
          null :
          Math.min(width, textWidth(targetWord, displayKey));
      const gapWidth = Math.max(12, refNatural || targetNatural || 12);
      return {
        ref: refNatural ?? gapWidth,
        target: targetNatural ?? gapWidth,
        gapWidth,
        newline: refWord === '\n' || targetWord === '\n',
      };
    });
    const packed = pack(measured, width),
          offsetByRow =
              new Map(indices.map((rowIndex, offset) => [rowIndex, offset])),
          result = {
            ...packed,
            ti,
            stage,
            kind,
            width,
            indices,
            offsetByRow,
            rows: allRows,
            height: Math.max(
                lineHeight(), (packed.lines.length - 1) * lineHeight()),
          };
    layouts.set(key, result);
    if (layouts.size > 16) layouts.delete(layouts.keys().next().value);
    return result;
  }
  function selectedRow(ti) {
    return config.getSelectedRow(ti);
  }
  function markup(layout, offset, side) {
    const rowIndex = layout.indices[offset], row = layout.rows[rowIndex],
          step = row[side];
    // A Steps-aligned pair may straddle the Thinking/Response boundary.
    // Its token belongs only in its captured phase, not both stage hosts.
    if (layout.kind !== 'input' && step != null &&
        config.phase(layout.ti, side, step) !== layout.stage)
      return `<span class="vCell" style="width:${
          layout.gapWidths[offset]}px" aria-hidden="true"></span>`;
    if (step == null) {
      if (layout.kind === 'input')
        return `<span class="vCell" style="width:${
            layout.gapWidths[offset]}px" aria-hidden="true"></span>`;
      const existingSide = side === 'ref' ? 'target' : 'ref',
            existing = row[existingSide],
            active = selectedRow(layout.ti)?.index === row.index;
      return `<span class="vCell gapAnchor" style="width:${
          layout.gapWidths[offset]}px" data-v-row="${
          rowIndex}"><button class="token missingOutput${
          active ? ' selected' : ''}" data-v-gap="${rowIndex}" data-v-turn="${
          layout.ti}" data-v-side="${side}" data-v-existing-side="${
          existingSide}" data-v-existing-step="${existing}" aria-label="No ${
          side === 'ref' ?
              'Reference' :
              'Target'} token · aligned position ${row.index}" aria-pressed="${
          active}"><span class="missingGlyph">∅</span></button></span>`;
    }
    const word = wordFor(layout.ti, layout.kind, side, step);
    const content = layout.kind === 'input' ?
        config.renderInputToken(word, step, side, layout.ti) :
        config.renderOutputToken(word, step, side, layout.ti);
    // Keep the existing space boxes for exact shaping, but remove the redundant
    // text wrapper.
    return content.replace('<span class="tokenText">', '')
        .replace('</span></button>', '</button>');
  }
  function paint(force = false, geometryOnly = false) {
    raf = 0;
    const sc = config.getScroller();
    if (!sc || painting) return;
    const width = Math.floor(sc.clientWidth),
          resized = lastWidth && width !== lastWidth;
    lastWidth = width;
    if (resized) {
      // A scroll event may run before ResizeObserver after a viewport change.
      // Restore the last painted token before the new geometry replaces it.
      const anchor = lastAnchor;
      if (anchor && anchor.kind !== 'absolute' && !geometryOnly) {
        restore(anchor);
        return;
      }
    }
    painting = true;
    const focused = sc.contains(document.activeElement) &&
            document.activeElement.closest('.virtualTextHost') ?
        document.activeElement :
        null;
    const viewport = sc.getBoundingClientRect(),
          hosts = [...sc.querySelectorAll('.virtualTextHost')].filter(
              (item) => item.clientWidth > 0);
    for (const item of hosts) {
      if (item.clientWidth) item._vLayout = layout(item);
    }
    const stageGroups = new Map();
    for (const item of hosts) {
      const l = item._vLayout;
      if (!l) continue;
      const key = [l.ti, l.stage, l.kind].join(':');
      if (!stageGroups.has(key)) stageGroups.set(key, []);
      stageGroups.get(key).push(item);
    }
    for (const group of stageGroups.values()) {
      const height = Math.max(...group.map((item) => item._vLayout.height));
      for (const item of group)
        if (item.style.height !== height + 'px')
          item.style.height = height + 'px';
    }
    if (geometryOnly) {
      painting = false;
      return;
    }
    const rects =
        new Map(hosts.map((item) => [item, item.getBoundingClientRect()]));
    for (const item of hosts) {
      const l = item._vLayout;
      if (!l) continue;
      const itemRect = rects.get(item);
      const top = viewport.top - itemRect.top,
            overscan = viewport.height * OVERSCAN;
      const batch = 4;
      const from = Math.max(
          0, Math.floor((top - overscan) / lineHeight() / batch) * batch);
      const to = Math.min(
          l.lines.length - 1,
          Math.ceil((top + viewport.height + overscan) / lineHeight() / batch) *
              batch,
      );
      const styleKey = [config.styleKey(), l.width].join(':');
      if (!item._vNodes || item._vLayoutIdentity !== l ||
          item.dataset.vStyle !== styleKey) {
        item.replaceChildren();
        item._vNodes = new Map();
        item._vLayoutIdentity = l;
        item.dataset.vStyle = styleKey;
      }
      const nodes = item._vNodes;
      for (const [line, node] of nodes)
        if (line < from || line >= to) {
          node.remove();
          nodes.delete(line);
        }
      const fragment = document.createDocumentFragment();
      for (let line = from; line < to; line++) {
        if (nodes.has(line)) continue;
        let cells = '';
        for (let offset = l.lines[line]; offset < l.lines[line + 1]; offset++)
          cells += markup(l, offset, item.dataset.vSide);
        const lineNode = document.createElement('span');
        lineNode.className = 'virtualTextLine';
        lineNode.dataset.vLine = line;
        lineNode.style.top = line * lineHeight() + 'px';
        lineNode.innerHTML = cells;
        nodes.set(line, lineNode);
        fragment.append(lineNode);
      }
      item.append(fragment);
    }
    if (focused && !focused.isConnected && focused.dataset.step != null) {
      sc.querySelector(
            `[data-turn="${focused.dataset.turn}"][data-step="${
                focused.dataset.step}"][data-side="${focused.dataset.side}"]`,
            )
          ?.focus({preventScroll: true});
    }
    painting = false;
    lastAnchor = capture(false);
  }
  function schedule() {
    if (!destroyed && !raf) raf = requestAnimationFrame(() => paint());
  }
  function findRow(ti, side, step) {
    return config.findRow(ti, side, step);
  }
  function locate(ti, side, step, stage) {
    const row = findRow(ti, side, step);
    if (!row) return null;
    const hostElement = config.getScroller()?.querySelector(
        `.virtualTextHost[data-v-kind="output"][data-v-side="target"][data-v-turn="${
            ti}"][data-v-stage="${stage}"]`,
    );
    const l = hostElement?._vLayout;
    if (!l) return null;
    const offset = l.offsetByRow.get(row.index);
    if (offset == null) return null;
    return {host: hostElement, layout: l, line: l.lineByOffset[offset]};
  }
  function jump(ti, step, side = 'target', pick = true) {
    if (!usesOutput(ti)) return false;
    const stage = config.phase(ti, side, step), sc = config.getScroller();
    paint(true);
    const found = locate(ti, side, step, stage);
    if (!found || !sc) return false;
    const viewport = sc.getBoundingClientRect(),
          top = config.getVisibleBounds?.()?.top ?? viewport.top + 60;
    sc.scrollTop += found.host.getBoundingClientRect().top +
        found.line * lineHeight() - top;
    paint(true);
    if (pick) config.onSelect(ti, step, side);
    paint(true);
    config.afterScroll?.();
    return true;
  }
  function jumpInput(ti, position, side = 'target') {
    if (!usesInput(ti)) return false;
    const sc = config.getScroller();
    paint(true);
    const hostElement = config.getScroller()?.querySelector(
              `.virtualTextHost[data-v-kind="input"][data-v-side="${
                  side}"][data-v-turn="${ti}"]`,
              ),
          l = hostElement?._vLayout;
    if (!sc || !l) return false;
    const rowIndex = l.rows.findIndex((row) => row[side] === position),
          offset = l.offsetByRow.get(rowIndex);
    if (offset == null) return false;
    const viewport = sc.getBoundingClientRect(),
          top = config.getVisibleBounds?.()?.top ?? viewport.top + 60;
    sc.scrollTop += hostElement.getBoundingClientRect().top +
        l.lineByOffset[offset] * lineHeight() - top;
    paint(true);
    config.afterScroll?.();
    return true;
  }
  function capture(repaint = true) {
    const sc = config.getScroller();
    if (!sc) return null;
    if (repaint) paint(true);
    const viewport = sc.getBoundingClientRect();
    const selected = [
      ...sc.querySelectorAll(
          '.virtualTextHost[data-v-kind="output"] .selected'),
    ].filter((node) => {
      const r = node.getBoundingClientRect();
      return r.bottom > viewport.top && r.top < viewport.bottom;
    });
    const selectedNode =
        selected.find((node) => node.dataset.side === 'target') || selected[0];
    if (selectedNode) {
      const hostElement = selectedNode.closest('.virtualTextHost'),
            rect = selectedNode.getBoundingClientRect(),
            lineNode = selectedNode.closest('.virtualTextLine');
      let side = selectedNode.dataset.side,
          step = Number(selectedNode.dataset.step);
      if (!Number.isInteger(step)) {
        side = selectedNode.dataset.vExistingSide;
        step = Number(selectedNode.dataset.vExistingStep);
      }
      if (Number.isInteger(step))
        return {
          kind: 'token',
          ti: +hostElement.dataset.vTurn,
          side,
          step,
          stage: hostElement.dataset.vStage,
          offset: rect.top - viewport.top,
          windowLine: lineNode ?
              +lineNode.dataset.vLine - (+hostElement.dataset.vStart || 0) :
              null,
        };
    }
    const visible = [];
    for (const hostElement of sc.querySelectorAll('.virtualTextHost')) {
      const l = hostElement._vLayout,
            rect = hostElement.getBoundingClientRect();
      if (!l || rect.bottom <= viewport.top || rect.top >= viewport.bottom)
        continue;
      const start = +hostElement.dataset.vStart || 0,
            line = Math.max(
                0,
                Math.min(
                    l.lines.length - 2,
                    start +
                        Math.floor(
                            Math.max(0, viewport.top - rect.top) /
                            lineHeight()),
                    ),
                ),
            offset = l.lines[line], rowIndex = l.indices[offset],
            row = l.rows[rowIndex],
            visibleOffset =
                rect.top + (line - start) * lineHeight() - viewport.top;
      if (l.kind === 'input')
        visible.push({
          kind: 'input',
          ti: l.ti,
          position: row.target ?? row.ref,
          stage: l.stage,
          offset: visibleOffset,
          windowLine: line - start,
        });
      else {
        const side = row.target != null ? 'target' : 'ref';
        visible.push({
          kind: 'token',
          ti: l.ti,
          side,
          step: row[side],
          stage: l.stage,
          offset: visibleOffset,
          windowLine: line - start,
        });
      }
    }
    return (
        visible.sort((a, b) => Math.abs(a.offset) - Math.abs(b.offset))[0] || {
          kind: 'absolute',
          scrollTop: sc.scrollTop,
        });
  }
  function restore(state) {
    const sc = config.getScroller();
    if (!sc || !state) return;
    paint(true, true);
    if (state.kind === 'absolute')
      sc.scrollTop = state.scrollTop;
    else if (state.kind === 'token') {
      const found = locate(state.ti, state.side, state.step, state.stage);
      if (found) {
        const viewport = sc.getBoundingClientRect(),
              bounds = config.getVisibleBounds?.(),
              offset = bounds ?
            Math.max(
                bounds.top - viewport.top,
                Math.min(
                    state.offset, bounds.bottom - viewport.top - lineHeight()),
                ) :
            state.offset;
        sc.scrollTop += found.host.getBoundingClientRect().top - viewport.top +
            found.line * lineHeight() - offset;
      }
    } else {
      const
          hostElement = config.getScroller()?.querySelector(
              `.virtualTextHost[data-v-kind="input"][data-v-side="target"][data-v-turn="${
                  state.ti}"]`,
              ),
          l = hostElement?._vLayout;
      if (l) {
        const rowIndex = l.rows.findIndex(
                  (row) => row.target === state.position ||
                      row.ref === state.position,
                  ),
              offset = l.offsetByRow.get(rowIndex);
        if (offset != null)
          sc.scrollTop += hostElement.getBoundingClientRect().top -
              sc.getBoundingClientRect().top +
              l.lineByOffset[offset] * lineHeight() - state.offset;
      }
    }
    paint(true);
    config.afterScroll?.();
  }
  function mount(anchor = null) {
    const next = config.getScroller();
    if (!next) return;
    if (scroller !== next) {
      if (scroller) {
        scroller.removeEventListener('scroll', schedule);
        scroller.removeEventListener('click', clickHandler);
      }
      scroller = next;
      scroller.addEventListener('scroll', schedule, {passive: true});
      clickHandler = (event) => {
        const gap = event.target.closest('[data-v-gap]');
        if (gap) {
          event.stopPropagation();
          jump(
              +gap.dataset.vTurn, +gap.dataset.vExistingStep,
              gap.dataset.vExistingSide, true);
          return;
        }
        const token = event.target.closest(
            '.virtualTextHost[data-v-kind="output"] button[data-step]',
        );
        if (token) {
          event.stopPropagation();
          jump(
              +token.dataset.turn, +token.dataset.step, token.dataset.side,
              true);
        }
      };
      scroller.addEventListener('click', clickHandler);
    }
    observer?.disconnect();
    observer = new ResizeObserver(() => {
      const nextWidth = Math.floor(scroller.clientWidth);
      if (nextWidth && nextWidth !== lastWidth) paint(true);
    });
    observer.observe(scroller);
    if (anchor)
      restore(anchor);
    else
      paint(true);
  }
  function clearLayout() {
    layouts.clear();
    widthCache.clear();
    textWidth.font = '';
    textWidth.scale = 1;
  }
  document.fonts?.ready?.then(() => {
    if (!destroyed) {
      clearLayout();
      schedule();
    }
  });
  function destroy() {
    destroyed = true;
    cancelAnimationFrame(raf);
    observer?.disconnect();
    scroller?.removeEventListener('scroll', schedule);
    scroller?.removeEventListener('click', clickHandler);
    layouts.clear();
    widthCache.clear();
  }
  return {
    usesOutput,
    usesInput,
    host,
    mount,
    paint,
    jump,
    jumpInput,
    capture,
    restore,
    clearLayout,
    destroy,
  };
}
