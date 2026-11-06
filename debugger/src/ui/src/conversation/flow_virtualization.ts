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
// Source SHA256: c9dd7226acd4626fddd9e2d6f3363f6ae182ce34a0d9a3c6c8ba1385f398ee14.
// Angular adapter: ES module, scoped lifecycle, cached resize anchors.
// Demo trend sampling is not bundled here; this component only renders text.

/** One aligned row: each side's step (or input position) in it, null for a gap. */
export interface FlowRow {
  index: number;
  ref: number | null;
  target: number | null;
}
/** One turn's words by side, and its rows by alignment. */
export interface FlowTurn {
  steps: {ref: string[]; target: string[]};
  inputTokens: {ref: string[]; target: string[]};
  alignments: Partial<Record<string, FlowRow[]>>;
  inputAlignments?: Partial<Record<string, FlowRow[]>>;
}
export type FlowAnchor =
  | {kind: 'absolute'; scrollTop: number}
  | {
      kind: 'token';
      ti: number;
      side: 'ref' | 'target';
      step: number;
      stage: 'Thinking' | 'Output';
      offset: number;
      windowLine?: number | null;
    }
  | {
      kind: 'input';
      ti: number;
      position: number;
      stage: 'Prefill';
      offset: number;
      windowLine?: number | null;
    };
export interface FlowConfig {
  threshold?: number;
  getTurn: (turn: number) => FlowTurn;
  getRows: (turn: number) => FlowRow[];
  getAlignment: () => string;
  isDebug: () => boolean;
  getScroller: () => HTMLElement | null;
  getVisibleBounds?: () => {top: number; bottom: number} | null;
  phase: (turn: number, side: string, index: number) => string;
  findRow: (turn: number, side: string, index: number) => FlowRow | undefined;
  getSelectedRow: (turn: number) => FlowRow | null;
  renderOutputToken: (word: string, index: number, side: string, turn: number) => string;
  renderInputToken: (word: string, index: number, side: string, turn: number) => string;
  displayKey: () => string;
  styleKey: () => string;
  widthAdjustment?: () => number;
  lineHeight?: () => number;
  onSelect: (turn: number, index: number, side: string) => void;
  afterScroll?: () => void;
}
export interface FlowVirtualizer {
  usesOutput: (turn: number) => boolean;
  usesInput: (turn: number) => boolean;
  host: (side: string, turn: number, stage: string, kind?: string) => string;
  mount: (anchor?: FlowAnchor | null) => void;
  paint: (geometryOnly?: boolean) => void;
  jump: (turn: number, index: number, side?: string, pick?: boolean) => boolean;
  jumpInput: (turn: number, position: number, side?: string) => boolean;
  capture: (repaint?: boolean) => FlowAnchor | null;
  restore: (anchor: FlowAnchor | null) => void;
  clearLayout: () => void;
  destroy: () => void;
}

type Side = 'ref' | 'target';
/** A host's text packed into lines at its width. */
interface FlowLayout {
  lines: number[];
  lineByOffset: Uint32Array;
  gapWidths: Float32Array;
  ti: number;
  stage: string;
  kind: string;
  width: number;
  indices: number[];
  offsetByRow: Map<number, number>;
  rows: FlowRow[];
  height: number;
}
interface MeasuredRow {
  ref: number;
  target: number;
  gapWidth: number;
  newline: boolean;
}
/** A host element with the layout it was painted with and its rendered lines. */
type FlowHost = HTMLElement & {
  vLayout?: FlowLayout;
  vNodes?: Map<number, HTMLElement>;
  vLayoutIdentity?: FlowLayout;
};

const OVERSCAN = 0.15;
export function createFlowVirtualizer(config: FlowConfig): FlowVirtualizer {
  const lineHeight = () => config.lineHeight?.() || 26;
  const threshold = config.threshold || 2048;
  const layouts = new Map<string, FlowLayout>();
  const widthCache = new Map<string, number>();
  const inputRowCache = new WeakMap<FlowTurn, FlowRow[]>();
  const rowVersions = new WeakMap<FlowRow[], number>();
  let nextRowVersion = 0;
  let raf = 0;
  let observer: ResizeObserver | null = null;
  let scroller: HTMLElement | null = null;
  let painting = false;
  let lastWidth = 0;
  let clickHandler: ((event: Event) => void) | null = null;
  let destroyed = false;
  let lastAnchor: FlowAnchor | null = null;
  // The text measure: a canvas in the probed font, scaled to the rendered width.
  let measureCanvas: HTMLCanvasElement | null = null;
  let measureFont = '';
  let measureScale = 1;

  function turn(ti: number) {
    return config.getTurn(ti);
  }
  function outputRows(ti: number) {
    return config.getRows(ti);
  }
  function inputTokens(ti: number, side: Side) {
    return turn(ti).inputTokens?.[side] || [];
  }
  function inputRows(ti: number): FlowRow[] {
    const t = turn(ti);
    const kind = config.getAlignment();
    if (t.inputAlignments?.[kind]) {
      return t.inputAlignments[kind];
    }
    const cached = inputRowCache.get(t);
    if (cached) {
      return cached;
    }
    const ref = inputTokens(ti, 'ref');
    const target = inputTokens(ti, 'target');
    const length = Math.max(ref.length, target.length);
    const rows = Array.from({length}, (_, index) => ({
      index,
      ref: index < ref.length ? index : null,
      target: index < target.length ? index : null,
    }));
    inputRowCache.set(t, rows);
    return rows;
  }
  function usesOutput(ti: number) {
    const t = turn(ti);
    return (
      Math.max(t.steps.ref.length, t.steps.target.length) > threshold &&
      !!t.alignments?.[config.getAlignment()]
    );
  }
  function usesInput(ti: number) {
    return Math.max(inputTokens(ti, 'ref').length, inputTokens(ti, 'target').length) > threshold;
  }
  function host(side: string, ti: number, stage: string, kind = 'output') {
    return `<span class="virtualTextHost" data-v-side="${side}" data-v-turn="${ti}" data-v-stage="${stage}" data-v-kind="${kind}"></span>`;
  }
  function rowIndices(ti: number, stage: string, kind: string) {
    const rows = kind === 'input' ? inputRows(ti) : outputRows(ti);
    if (kind === 'input') {
      return rows.map((_, i) => i);
    }
    return rows.reduce<number[]>((indices, row, index) => {
      for (const side of ['ref', 'target'] as const) {
        const step = row[side];
        if (step != null && config.phase(ti, side, step) === stage) {
          indices.push(index);
          break;
        }
      }
      return indices;
    }, []);
  }
  function wordFor(ti: number, kind: string, side: Side, index: number) {
    return kind === 'input' ? inputTokens(ti, side)[index] : turn(ti).steps[side][index];
  }
  function textWidth(text: string, displayKey: string) {
    const key = displayKey + '\u0000' + text;
    const cached = widthCache.get(key);
    if (cached !== undefined) {
      return cached;
    }
    if (/^[\r\n\u200b-\u200d\ufeff]*$/.test(text)) {
      widthCache.set(key, 10);
      return 10;
    }
    const canvas = measureCanvas || (measureCanvas = document.createElement('canvas'));
    const context = canvas.getContext('2d');
    if (!context) {
      return 10;
    }
    if (!measureFont) {
      const probe = document.createElement('span');
      probe.className = 'virtualTextWidthProbe';
      probe.textContent = 'mmmmmmmm iiii looks';
      document.body.append(probe);
      const style = getComputedStyle(probe);
      measureFont = style.font;
      context.font = measureFont;
      measureScale =
        probe.getBoundingClientRect().width /
        Math.max(1, context.measureText(probe.textContent).width);
      probe.remove();
    }
    context.font = measureFont;
    const value = String(text);
    const spaces = (value.match(/ /g) || []).length;
    const plain = value.replaceAll(' ', '');
    const width =
      Math.max(1, context.measureText(plain).width * (measureScale || 1) + spaces * 9.1) +
      (config.widthAdjustment?.() || 0);
    widthCache.set(key, width);
    return width;
  }
  function pack(rows: MeasuredRow[], width: number, offset = 0) {
    const lines = [offset];
    const lineByOffset = new Uint32Array(rows.length);
    const gapWidths = new Float32Array(rows.length);
    let refUsed = 0;
    let targetUsed = 0;
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      if (
        (refUsed + row.ref > width || targetUsed + row.target > width) &&
        (refUsed > 0 || targetUsed > 0)
      ) {
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
  function layout(hostElement: HTMLElement): FlowLayout {
    const ti = Number(hostElement.dataset['vTurn'] ?? 0);
    const stage = hostElement.dataset['vStage'] ?? '';
    const kind = hostElement.dataset['vKind'] ?? '';
    const allRows = kind === 'input' ? inputRows(ti) : outputRows(ti);
    if (!rowVersions.has(allRows)) {
      rowVersions.set(allRows, ++nextRowVersion);
    }
    const width = Math.max(1, Math.floor(hostElement.clientWidth));
    const displayKey = config.displayKey();
    const key = [
      ti,
      stage,
      kind,
      width,
      rowVersions.get(allRows),
      kind === 'input' ? '' : config.getAlignment(),
      kind === 'input' ? '' : config.isDebug(),
      displayKey,
    ].join(':');
    const cached = layouts.get(key);
    if (cached) {
      layouts.delete(key);
      layouts.set(key, cached);
      return cached;
    }
    const indices = rowIndices(ti, stage, kind);
    const measured = indices.map((rowIndex) => {
      const row = allRows[rowIndex];
      const refWord =
        row.ref == null || (kind !== 'input' && config.phase(ti, 'ref', row.ref) !== stage)
          ? null
          : wordFor(ti, kind, 'ref', row.ref);
      const targetWord =
        row.target == null ||
        (kind !== 'input' && config.phase(ti, 'target', row.target) !== stage)
          ? null
          : wordFor(ti, kind, 'target', row.target);
      const refNatural = refWord == null ? null : Math.min(width, textWidth(refWord, displayKey));
      const targetNatural =
        targetWord == null ? null : Math.min(width, textWidth(targetWord, displayKey));
      const gapWidth = Math.max(12, refNatural || targetNatural || 12);
      return {
        ref: refNatural ?? gapWidth,
        target: targetNatural ?? gapWidth,
        gapWidth,
        newline: refWord === '\n' || targetWord === '\n',
      };
    });
    const packed = pack(measured, width);
    const offsetByRow = new Map(indices.map((rowIndex, offset) => [rowIndex, offset]));
    const result = {
      ...packed,
      ti,
      stage,
      kind,
      width,
      indices,
      offsetByRow,
      rows: allRows,
      height: Math.max(lineHeight(), (packed.lines.length - 1) * lineHeight()),
    };
    layouts.set(key, result);
    if (layouts.size > 16) {
      const oldest = layouts.keys().next().value;
      if (oldest !== undefined) {
        layouts.delete(oldest);
      }
    }
    return result;
  }
  function selectedRow(ti: number) {
    return config.getSelectedRow(ti);
  }
  function markup(layout: FlowLayout, offset: number, side: Side) {
    const rowIndex = layout.indices[offset];
    const row = layout.rows[rowIndex];
    const step = row[side];
    // A Steps-aligned pair may straddle the Thinking/Response boundary.
    // Its token belongs only in its captured phase, not both stage hosts.
    if (
      layout.kind !== 'input' &&
      step != null &&
      config.phase(layout.ti, side, step) !== layout.stage
    ) {
      return `<span class="vCell" style="width:${layout.gapWidths[offset]}px" aria-hidden="true"></span>`;
    }
    if (step == null) {
      if (layout.kind === 'input') {
        return `<span class="vCell" style="width:${layout.gapWidths[offset]}px" aria-hidden="true"></span>`;
      }
      const existingSide = side === 'ref' ? 'target' : 'ref';
      const existing = row[existingSide];
      const active = selectedRow(layout.ti)?.index === row.index;
      return `<span class="vCell gapAnchor" style="width:${layout.gapWidths[offset]}px" data-v-row="${rowIndex}"><button class="token missingOutput${active ? ' selected' : ''}" data-v-gap="${rowIndex}" data-v-turn="${layout.ti}" data-v-side="${side}" data-v-existing-side="${existingSide}" data-v-existing-step="${existing}" aria-label="No ${side === 'ref' ? 'Reference' : 'Target'} token · aligned position ${row.index}" aria-pressed="${active}"><span class="missingGlyph">∅</span></button></span>`;
    }
    const word = wordFor(layout.ti, layout.kind, side, step);
    const content =
      layout.kind === 'input'
        ? config.renderInputToken(word, step, side, layout.ti)
        : config.renderOutputToken(word, step, side, layout.ti);
    // Keep the existing space boxes for exact shaping, but remove the redundant text wrapper.
    return content.replace('<span class="tokenText">', '').replace('</span></button>', '</button>');
  }
  function paint(geometryOnly = false) {
    raf = 0;
    const sc = config.getScroller();
    if (!sc || painting) {
      return;
    }
    const width = Math.floor(sc.clientWidth);
    const resized = lastWidth && width !== lastWidth;
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
    const active = document.activeElement as HTMLElement | null;
    const focused = sc.contains(active) && active?.closest('.virtualTextHost') ? active : null;
    const viewport = sc.getBoundingClientRect();
    const hosts = Array.from(sc.querySelectorAll<FlowHost>('.virtualTextHost')).filter(
      (item) => item.clientWidth > 0,
    );
    for (const item of hosts) {
      if (item.clientWidth) {
        item.vLayout = layout(item);
      }
    }
    const stageGroups = new Map<string, FlowHost[]>();
    for (const item of hosts) {
      const l = item.vLayout;
      if (!l) {
        continue;
      }
      const key = [l.ti, l.stage, l.kind].join(':');
      let group = stageGroups.get(key);
      if (!group) {
        group = [];
        stageGroups.set(key, group);
      }
      group.push(item);
    }
    for (const group of stageGroups.values()) {
      const height = Math.max(...group.map((item) => item.vLayout?.height ?? 0));
      for (const item of group) {
        if (item.style.height !== height + 'px') {
          item.style.height = height + 'px';
        }
      }
    }
    if (geometryOnly) {
      painting = false;
      return;
    }
    const rects = new Map(hosts.map((item) => [item, item.getBoundingClientRect()]));
    for (const item of hosts) {
      const l = item.vLayout;
      const itemRect = rects.get(item);
      if (!l || !itemRect) {
        continue;
      }
      const top = viewport.top - itemRect.top;
      const overscan = viewport.height * OVERSCAN;
      const batch = 4;
      const from = Math.max(0, Math.floor((top - overscan) / lineHeight() / batch) * batch);
      const to = Math.min(
        l.lines.length - 1,
        Math.ceil((top + viewport.height + overscan) / lineHeight() / batch) * batch,
      );
      const styleKey = [config.styleKey(), l.width].join(':');
      if (!item.vNodes || item.vLayoutIdentity !== l || item.dataset['vStyle'] !== styleKey) {
        item.replaceChildren();
        item.vNodes = new Map();
        item.vLayoutIdentity = l;
        item.dataset['vStyle'] = styleKey;
      }
      const nodes = item.vNodes;
      for (const [line, node] of nodes) {
        if (line < from || line >= to) {
          node.remove();
          nodes.delete(line);
        }
      }
      const fragment = document.createDocumentFragment();
      for (let line = from; line < to; line++) {
        if (nodes.has(line)) {
          continue;
        }
        let cells = '';
        for (let offset = l.lines[line]; offset < l.lines[line + 1]; offset++) {
          cells += markup(l, offset, item.dataset['vSide'] as Side);
        }
        const lineNode = document.createElement('span');
        lineNode.className = 'virtualTextLine';
        lineNode.dataset['vLine'] = String(line);
        lineNode.style.top = line * lineHeight() + 'px';
        lineNode.innerHTML = cells;
        nodes.set(line, lineNode);
        fragment.append(lineNode);
      }
      item.append(fragment);
    }
    if (focused && !focused.isConnected && focused.dataset['step'] != null) {
      sc.querySelector<HTMLElement>(
        `[data-turn="${focused.dataset['turn']}"][data-step="${focused.dataset['step']}"][data-side="${focused.dataset['side']}"]`,
      )?.focus({preventScroll: true});
    }
    painting = false;
    lastAnchor = capture(false);
  }
  function schedule() {
    if (!destroyed && !raf) {
      raf = requestAnimationFrame(() => paint());
    }
  }
  function findRow(ti: number, side: string, step: number) {
    return config.findRow(ti, side, step);
  }
  function locate(ti: number, side: string, step: number, stage: string) {
    const row = findRow(ti, side, step);
    if (!row) {
      return null;
    }
    const hostElement = config
      .getScroller()
      ?.querySelector<FlowHost>(
        `.virtualTextHost[data-v-kind="output"][data-v-side="target"][data-v-turn="${ti}"][data-v-stage="${stage}"]`,
      );
    const l = hostElement?.vLayout;
    if (!hostElement || !l) {
      return null;
    }
    const offset = l.offsetByRow.get(row.index);
    if (offset == null) {
      return null;
    }
    return {host: hostElement, layout: l, line: l.lineByOffset[offset]};
  }
  function jump(ti: number, step: number, side = 'target', pick = true) {
    if (!usesOutput(ti)) {
      return false;
    }
    const stage = config.phase(ti, side, step);
    const sc = config.getScroller();
    paint();
    const found = locate(ti, side, step, stage);
    if (!found || !sc) {
      return false;
    }
    const viewport = sc.getBoundingClientRect();
    const top = config.getVisibleBounds?.()?.top ?? viewport.top + 60;
    sc.scrollTop += found.host.getBoundingClientRect().top + found.line * lineHeight() - top;
    paint();
    if (pick) {
      config.onSelect(ti, step, side);
    }
    paint();
    config.afterScroll?.();
    return true;
  }
  function jumpInput(ti: number, position: number, side = 'target') {
    if (!usesInput(ti)) {
      return false;
    }
    const sc = config.getScroller();
    paint();
    const hostElement = config
      .getScroller()
      ?.querySelector<FlowHost>(
        `.virtualTextHost[data-v-kind="input"][data-v-side="${side}"][data-v-turn="${ti}"]`,
      );
    const l = hostElement?.vLayout;
    if (!sc || !hostElement || !l) {
      return false;
    }
    const rowIndex = l.rows.findIndex((row) => row[side as Side] === position);
    const offset = l.offsetByRow.get(rowIndex);
    if (offset == null) {
      return false;
    }
    const viewport = sc.getBoundingClientRect();
    const top = config.getVisibleBounds?.()?.top ?? viewport.top + 60;
    sc.scrollTop +=
      hostElement.getBoundingClientRect().top + l.lineByOffset[offset] * lineHeight() - top;
    paint();
    config.afterScroll?.();
    return true;
  }
  function capture(repaint = true): FlowAnchor | null {
    const sc = config.getScroller();
    if (!sc) {
      return null;
    }
    if (repaint) {
      paint();
    }
    const viewport = sc.getBoundingClientRect();
    const selected = Array.from(
      sc.querySelectorAll<HTMLElement>('.virtualTextHost[data-v-kind="output"] .selected'),
    ).filter((node) => {
      const r = node.getBoundingClientRect();
      return r.bottom > viewport.top && r.top < viewport.bottom;
    });
    const selectedNode = selected.find((node) => node.dataset['side'] === 'target') || selected[0];
    if (selectedNode) {
      const hostElement = selectedNode.closest<HTMLElement>('.virtualTextHost');
      const rect = selectedNode.getBoundingClientRect();
      const lineNode = selectedNode.closest<HTMLElement>('.virtualTextLine');
      let side = selectedNode.dataset['side'];
      let step = Number(selectedNode.dataset['step']);
      if (!Number.isInteger(step)) {
        side = selectedNode.dataset['vExistingSide'];
        step = Number(selectedNode.dataset['vExistingStep']);
      }
      if (hostElement && Number.isInteger(step)) {
        return {
          kind: 'token',
          ti: Number(hostElement.dataset['vTurn'] ?? 0),
          side: side as Side,
          step,
          stage: hostElement.dataset['vStage'] as 'Thinking' | 'Output',
          offset: rect.top - viewport.top,
          windowLine: lineNode
            ? Number(lineNode.dataset['vLine'] ?? 0) - Number(hostElement.dataset['vStart'] || 0)
            : null,
        };
      }
    }
    const visible: FlowAnchor[] = [];
    for (const hostElement of Array.from(sc.querySelectorAll<FlowHost>('.virtualTextHost'))) {
      const l = hostElement.vLayout;
      const rect = hostElement.getBoundingClientRect();
      if (!l || rect.bottom <= viewport.top || rect.top >= viewport.bottom) {
        continue;
      }
      const start = Number(hostElement.dataset['vStart'] || 0);
      const line = Math.max(
        0,
        Math.min(
          l.lines.length - 2,
          start + Math.floor(Math.max(0, viewport.top - rect.top) / lineHeight()),
        ),
      );
      const offset = l.lines[line];
      const rowIndex = l.indices[offset];
      const row = l.rows[rowIndex];
      const visibleOffset = rect.top + (line - start) * lineHeight() - viewport.top;
      if (l.kind === 'input') {
        const position = row.target ?? row.ref;
        if (position !== null) {
          visible.push({
            kind: 'input',
            ti: l.ti,
            position,
            stage: l.stage as 'Prefill',
            offset: visibleOffset,
            windowLine: line - start,
          });
        }
      } else {
        const side = row.target != null ? 'target' : 'ref';
        const step = row[side];
        if (step !== null) {
          visible.push({
            kind: 'token',
            ti: l.ti,
            side,
            step,
            stage: l.stage as 'Thinking' | 'Output',
            offset: visibleOffset,
            windowLine: line - start,
          });
        }
      }
    }
    return (
      visible.sort(
        (a, b) =>
          Math.abs((a as {offset: number}).offset) - Math.abs((b as {offset: number}).offset),
      )[0] || {
        kind: 'absolute',
        scrollTop: sc.scrollTop,
      }
    );
  }
  function restore(state: FlowAnchor | null) {
    const sc = config.getScroller();
    if (!sc || !state) {
      return;
    }
    paint(true);
    if (state.kind === 'absolute') {
      sc.scrollTop = state.scrollTop;
    } else if (state.kind === 'token') {
      const found = locate(state.ti, state.side, state.step, state.stage);
      if (found) {
        const viewport = sc.getBoundingClientRect();
        const bounds = config.getVisibleBounds?.();
        const offset = bounds
          ? Math.max(
              bounds.top - viewport.top,
              Math.min(state.offset, bounds.bottom - viewport.top - lineHeight()),
            )
          : state.offset;
        sc.scrollTop +=
          found.host.getBoundingClientRect().top -
          viewport.top +
          found.line * lineHeight() -
          offset;
      }
    } else {
      const hostElement = config
        .getScroller()
        ?.querySelector<FlowHost>(
          `.virtualTextHost[data-v-kind="input"][data-v-side="target"][data-v-turn="${state.ti}"]`,
        );
      const l = hostElement?.vLayout;
      if (hostElement && l) {
        const rowIndex = l.rows.findIndex(
          (row) => row.target === state.position || row.ref === state.position,
        );
        const offset = l.offsetByRow.get(rowIndex);
        if (offset != null) {
          sc.scrollTop +=
            hostElement.getBoundingClientRect().top -
            sc.getBoundingClientRect().top +
            l.lineByOffset[offset] * lineHeight() -
            state.offset;
        }
      }
    }
    paint();
    config.afterScroll?.();
  }
  function mount(anchor: FlowAnchor | null = null) {
    const next = config.getScroller();
    if (!next) {
      return;
    }
    if (scroller !== next) {
      if (scroller && clickHandler) {
        scroller.removeEventListener('scroll', schedule);
        scroller.removeEventListener('click', clickHandler);
      }
      scroller = next;
      scroller.addEventListener('scroll', schedule, {passive: true});
      clickHandler = (event) => {
        const target = event.target as Element;
        const gap = target.closest<HTMLElement>('[data-v-gap]');
        if (gap) {
          event.stopPropagation();
          config.onSelect(
            Number(gap.dataset['vTurn'] ?? 0),
            Number(gap.dataset['vExistingStep'] ?? 0),
            gap.dataset['vExistingSide'] ?? 'target',
          );
          paint();
          return;
        }
        const token = target.closest<HTMLElement>(
          '.virtualTextHost[data-v-kind="output"] button[data-step]',
        );
        if (token) {
          event.stopPropagation();
          config.onSelect(
            Number(token.dataset['turn'] ?? 0),
            Number(token.dataset['step'] ?? 0),
            token.dataset['side'] ?? 'target',
          );
          paint();
        }
      };
      scroller.addEventListener('click', clickHandler);
    }
    observer?.disconnect();
    const activeScroller = scroller;
    observer = new ResizeObserver(() => {
      const nextWidth = Math.floor(activeScroller.clientWidth);
      if (nextWidth && nextWidth !== lastWidth) {
        paint();
      }
    });
    observer.observe(activeScroller);
    if (anchor) {
      restore(anchor);
    } else {
      paint();
    }
  }
  function clearLayout() {
    layouts.clear();
    widthCache.clear();
    measureFont = '';
    measureScale = 1;
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
    if (clickHandler) {
      scroller?.removeEventListener('click', clickHandler);
    }
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
