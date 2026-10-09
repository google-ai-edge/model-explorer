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

import dagre from '@dagrejs/dagre';
import type {
  Anchor,
  ComparisonRow,
  Semantic,
} from '../../../data/contracts/types';
import {formatMetric} from '../../../shared/format/format';

export type ArchitectureAction =
  | 'focus'
  | 'fit'
  | 'readable'
  | 'zoomIn'
  | 'zoomOut';
export interface ArchitectureSelection {
  kind: 'architecture' | 'tensor';
  id: string;
}
export interface ArchitectureOptions {
  model: Semantic;
  layer: number;
  rows: ComparisonRow[];
  metric: string;
  selection: ArchitectureSelection | null;
}
interface CanvasNode {
  id: string;
  kind: 'architecture' | 'tensor';
  name: string;
  namespace?: string;
  port?: boolean;
  compound?: boolean;
  executable?: boolean;
  anchor?: Anchor;
  width: number;
  height: number;
}
interface CanvasEdge {
  id: string;
  source: string;
  target: string;
}
export interface ArchitectureTopology {
  nodes: CanvasNode[];
  edges: CanvasEdge[];
}
interface NodeElement {
  node: CanvasNode;
  group: SVGGElement;
  x: number;
  y: number;
}
export interface ArchitectureViewport {
  layer: number;
  zoom: number;
  left: number;
  top: number;
}
type Viewport = Omit<ArchitectureViewport, 'layer'>;
const NS = 'http://www.w3.org/2000/svg';
const MIN_ZOOM = 0.15,
  MAX_ZOOM = 3;
let serial = 0;

function svgElement<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attributes: Record<string, string | number> = {},
  text?: string,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attributes))
    node.setAttribute(key, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}
function binding(of: string): {node: string; port: string} {
  const split = of.lastIndexOf(':');
  if (split < 1 || split === of.length - 1)
    throw new Error(`Invalid anchor binding: ${of}`);
  return {node: of.slice(0, split), port: of.slice(split + 1)};
}
const portKey = (node: string, port: string) => JSON.stringify([node, port]);

/** Anchors bind exact ports; no module names or implicit output 0 are synthesized. */
export function buildArchitectureTopology(
  model: Semantic,
  layer: number,
): ArchitectureTopology | null {
  const info = model.layers[layer],
    definition = model.semantic_graph[info?.def];
  if (!definition) return null;
  const nodes: CanvasNode[] = [],
    edges: CanvasEdge[] = [];
  const known = new Set([
    ...definition.inputs.map((input) => input.id),
    ...definition.nodes.map((node) => node.id),
  ]);
  const anchoredOutputs = new Map<string, string[]>(),
    anchoredInputs = new Map<string, string[]>();
  const inputAliases = new Map<string, string>();
  for (const input of definition.inputs) {
    const anchors = definition.anchors.filter(
      (anchor) => binding(anchor.of).node === input.id && anchor.edge !== 'in',
    );
    // A layer-input anchor and its input port represent the same value.
    if (anchors.length === 1)
      inputAliases.set(input.id, 'anchor:' + anchors[0].id);
    else
      nodes.push({
        id: input.id,
        kind: 'architecture',
        name: definition.inputs.length === 1 ? 'Layer input' : input.id,
        port: true,
        width: 178,
        height: 32,
      });
  }
  for (const node of definition.nodes) {
    const declaredOps = (node as typeof node & {ops?: unknown[]}).ops;
    const compound =
      Math.max(
        declaredOps?.length ?? 0,
        Object.keys(info.attrs?.[node.id]?.ops ?? {}).length,
      ) > 1;
    nodes.push({
      id: node.id,
      kind: 'architecture',
      name: node.label || node.id,
      namespace: node.namespace,
      compound,
      executable: true,
      width: 208,
      height: compound ? 52 : 34,
    });
  }
  for (const anchor of definition.anchors) {
    const source = binding(anchor.of),
      id = 'anchor:' + anchor.id;
    if (!known.has(source.node))
      throw new Error(`Unresolved anchor: ${anchor.of}`);
    nodes.push({
      id,
      kind: 'tensor',
      name: anchor.label || anchor.semantic || anchor.id,
      anchor,
      width: 244,
      height: 44,
    });
    const index = anchor.edge === 'in' ? anchoredInputs : anchoredOutputs;
    const key = portKey(source.node, source.port);
    index.set(key, [...(index.get(key) ?? []), id]);
    if (inputAliases.get(source.node) !== id)
      edges.push({
        id: 'anchor:' + anchor.id,
        source:
          anchor.edge === 'in'
            ? id
            : (inputAliases.get(source.node) ?? source.node),
        target:
          anchor.edge === 'in'
            ? (inputAliases.get(source.node) ?? source.node)
            : id,
      });
  }
  for (const node of definition.nodes) {
    for (const [index, edge] of (node.incomingEdges ?? []).entries()) {
      if (!known.has(edge.sourceNodeId))
        throw new Error(`Unresolved edge: ${edge.sourceNodeId} → ${node.id}`);
      const sources = anchoredOutputs.get(
        portKey(edge.sourceNodeId, edge.sourceNodeOutputId),
      ) ?? [inputAliases.get(edge.sourceNodeId) ?? edge.sourceNodeId];
      const targets = anchoredInputs.get(
        portKey(node.id, edge.targetNodeInputId),
      ) ?? [node.id];
      for (const source of sources)
        for (const target of targets)
          edges.push({
            id: `${node.id}:${index}:${source}:${target}`,
            source,
            target,
          });
    }
  }
  if (new Set(nodes.map((node) => node.id)).size !== nodes.length)
    throw new Error('Duplicate semantic graph node or anchor ID.');
  return nodes.length ? {nodes, edges} : null;
}

export function architectureMetric(
  row: ComparisonRow | undefined,
  metric: string,
): number | null {
  const reading = row?.metrics[metric];
  return reading?.status === 'ok' &&
    typeof reading.value === 'number' &&
    Number.isFinite(reading.value)
    ? reading.value
    : null;
}
function magnitude(value: number | null, metric: string): number | null {
  return value === null
    ? null
    : metric === 'CosSim'
      ? Math.max(0, 1 - value)
      : Math.abs(value);
}
const format = (value: number | null) =>
  formatMetric(value, {
    placeholder: '—',
    exponentDigits: 3,
    small: 0.0001,
    large: 1000,
  });

export class ArchitectureRenderer {
  private options?: ArchitectureOptions;
  private readonly marker = 'architecture-arrow-' + ++serial;
  private readonly listeners = new AbortController();
  private viewportListeners?: AbortController;
  private model?: Semantic;
  private key = '';
  private currentLayer: number | null = null;
  private readonly viewByLayer = new Map<number, Viewport>();
  private zoom = 1;
  private width = 0;
  private height = 0;
  private svg?: SVGSVGElement;
  private viewport?: HTMLDivElement;
  private elements = new Map<string, NodeElement>();
  private edges: {source: string; target: string; path: SVGPathElement}[] = [];
  private controls = new Map<ArchitectureAction, HTMLButtonElement>();

  constructor(
    private readonly host: HTMLElement,
    private readonly actions: {
      inspect(id: string): void;
      execute(id: string): void;
      viewportChanged?(value: ArchitectureViewport): void;
    },
  ) {
    const activate = (event: Event) => {
      const target = event.target instanceof Element ? event.target : null;
      const execute = target?.closest<SVGGElement>(
        '[data-architecture-execute]',
      );
      if (execute && host.contains(execute)) {
        this.actions.execute(execute.dataset['architectureExecute']!);
        return;
      }
      const group = target?.closest<SVGGElement>('[data-architecture-id]');
      if (!group || !host.contains(group)) return;
      const item = this.elements.get(group.dataset['architectureId']!);
      if (item) this.actions.inspect(item.node.id);
    };
    host.addEventListener('click', activate, {signal: this.listeners.signal});
    host.addEventListener(
      'keydown',
      (event) => {
        if (
          (event.key === 'Enter' || event.key === ' ') &&
          event.target instanceof Element &&
          event.target.closest(
            '[data-architecture-id],[data-architecture-execute]',
          )
        ) {
          event.preventDefault();
          activate(event);
        }
      },
      {signal: this.listeners.signal},
    );
  }

  render(options: ArchitectureOptions): void {
    this.options = options;
    if (this.model !== options.model) {
      this.viewByLayer.clear();
      this.currentLayer = null;
      this.key = '';
      this.model = options.model;
    }
    try {
      const topology = buildArchitectureTopology(options.model, options.layer);
      const key = JSON.stringify([options.layer, topology]);
      if (key !== this.key) {
        this.saveViewport();
        const position = this.viewByLayer.get(options.layer) ?? {
          zoom: 1,
          left: 0,
          top: 0,
        };
        this.currentLayer = options.layer;
        this.zoom = position.zoom;
        if (topology) {
          this.draw(topology);
          this.applyDimensions();
          this.viewport!.scrollLeft = position.left;
          this.viewport!.scrollTop = position.top;
        } else
          this.empty('No architecture definition captured for this layer.');
        this.key = key;
      }
      if (topology) this.update();
    } catch (error) {
      this.empty(
        error instanceof Error
          ? error.message
          : 'Architecture could not be displayed.',
        true,
      );
      this.key = '';
    }
  }

  control(action: ArchitectureAction, focusId?: string): boolean {
    if (!this.svg || !this.viewport) return false;
    if (action === 'zoomIn') this.setZoom(this.zoom * 1.2);
    else if (action === 'zoomOut') this.setZoom(this.zoom / 1.2);
    else if (action === 'readable') this.setZoom(1);
    else if (action === 'fit') {
      this.setZoom(
        Math.min(
          1,
          (this.viewport.clientWidth - 48) / this.width,
          (this.viewport.clientHeight - 80) / this.height,
        ),
      );
      this.center(this.width / 2, this.height / 2);
    } else if (action === 'focus') {
      const item = this.elements.get(this.selectedId(focusId) ?? '');
      if (!item) return false;
      this.center(item.x, item.y);
    } else return false;
    return true;
  }

  destroy(): void {
    this.listeners.abort();
    this.viewportListeners?.abort();
    this.host.replaceChildren();
    this.elements.clear();
    this.viewByLayer.clear();
  }

  restoreViewport(value: ArchitectureViewport): void {
    if (value.layer !== this.currentLayer || !this.viewport) return;
    this.zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, value.zoom));
    this.applyDimensions();
    this.updateControls();
    this.viewport.scrollLeft = Math.max(0, value.left);
    this.viewport.scrollTop = Math.max(0, value.top);
    this.saveViewport();
  }

  private selectedId(focusId?: string): string | null {
    const selection = this.options?.selection;
    let id = focusId ?? selection?.id;
    if (!id) return null;
    if (!focusId && selection?.kind === 'tensor' && !id.startsWith('anchor:'))
      id = 'anchor:' + id;
    if (this.elements.has(id)) return id;
    return this.elements.has('anchor:' + id) ? 'anchor:' + id : null;
  }
  private saveViewport(): void {
    if (this.currentLayer !== null && this.viewport) {
      this.viewByLayer.set(this.currentLayer, {
        zoom: this.zoom,
        left: this.viewport.scrollLeft,
        top: this.viewport.scrollTop,
      });
      this.actions.viewportChanged?.({
        layer: this.currentLayer,
        ...this.viewByLayer.get(this.currentLayer)!,
      });
    }
  }
  private applyDimensions(): void {
    this.svg?.setAttribute('width', String(this.width * this.zoom));
    this.svg?.setAttribute('height', String(this.height * this.zoom));
  }
  private setZoom(value: number): void {
    if (!this.svg || !this.viewport) return;
    const viewport = this.viewport,
      bounds = viewport.getBoundingClientRect(),
      before = this.svg.getBoundingClientRect();
    const x =
      (bounds.left + viewport.clientWidth / 2 - before.left) / this.zoom;
    const y = (bounds.top + viewport.clientHeight / 2 - before.top) / this.zoom;
    this.zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, value));
    this.applyDimensions();
    this.center(x, y);
    this.updateControls();
  }
  private center(x: number, y: number): void {
    if (!this.svg || !this.viewport) return;
    const viewport = this.viewport,
      bounds = viewport.getBoundingClientRect(),
      svg = this.svg.getBoundingClientRect();
    viewport.scrollLeft +=
      svg.left + x * this.zoom - bounds.left - viewport.clientWidth / 2;
    viewport.scrollTop +=
      svg.top + y * this.zoom - bounds.top - viewport.clientHeight / 2;
    this.saveViewport();
  }
  private updateControls(): void {
    const reset = this.controls.get('readable');
    if (!reset) return;
    reset.textContent = Math.round(this.zoom * 100) + '%';
    reset.setAttribute(
      'aria-label',
      `Zoom ${Math.round(this.zoom * 100)} percent. Reset zoom to 100 percent`,
    );
    this.controls.get('zoomOut')!.disabled = this.zoom <= MIN_ZOOM;
    this.controls.get('zoomIn')!.disabled = this.zoom >= MAX_ZOOM;
    this.controls.get('focus')!.disabled = !this.selectedId();
  }
  private createControls(): HTMLDivElement {
    const controls = document.createElement('div');
    controls.className = 'architectureControls';
    controls.setAttribute('role', 'group');
    controls.setAttribute('aria-label', 'Architecture canvas controls');
    this.controls.clear();
    const buttons: [ArchitectureAction, string, string][] = [
      ['zoomOut', 'remove', 'Zoom out'],
      ['readable', '', 'Reset zoom to 100%'],
      ['zoomIn', 'add', 'Zoom in'],
      ['fit', 'fit_screen', 'Fit architecture'],
      ['focus', 'center_focus_strong', 'Focus selected node'],
    ];
    for (const [action, icon, label] of buttons) {
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset['architectureControl'] = action;
      button.title = label;
      button.setAttribute('aria-label', label);
      if (icon) {
        const symbol = document.createElement('span');
        symbol.className = 'architectureIcon';
        symbol.setAttribute('aria-hidden', 'true');
        symbol.textContent = icon;
        button.append(symbol);
      } else {
        button.className = 'architectureZoom';
        button.textContent = '100%';
      }
      button.addEventListener('click', () => this.control(action));
      this.controls.set(action, button);
      controls.append(button);
    }
    return controls;
  }

  private draw(topology: ArchitectureTopology): void {
    const graph = new dagre.graphlib.Graph({multigraph: true});
    graph.setGraph({
      rankdir: 'TB',
      nodesep: 30,
      ranksep: 18,
      marginx: 24,
      marginy: 20,
    });
    graph.setDefaultEdgeLabel(() => ({}));
    for (const node of topology.nodes)
      graph.setNode(node.id, {width: node.width, height: node.height});
    for (const edge of topology.edges)
      graph.setEdge(edge.source, edge.target, {}, edge.id);
    if (!dagre.graphlib.alg.isAcyclic(graph))
      throw new Error('Semantic graph contains a cycle.');
    dagre.layout(graph);
    this.width = graph.graph().width ?? 0;
    this.height = graph.graph().height ?? 0;
    const svg = svgElement('svg', {
      class: 'architectureSvg',
      width: this.width,
      height: this.height,
      viewBox: `0 0 ${this.width} ${this.height}`,
      role: 'group',
      'aria-label': `Layer ${this.options!.layer} architecture graph`,
    });
    const defs = svgElement('defs');
    for (const suffix of ['', '-relation']) {
      const marker = svgElement('marker', {
        id: this.marker + suffix,
        viewBox: '0 0 8 8',
        refX: 7,
        refY: 4,
        markerWidth: 6,
        markerHeight: 6,
        orient: 'auto',
      });
      marker.append(
        svgElement('path', {
          d: 'M0,0 L8,4 L0,8',
          class: suffix
            ? 'architectureRelationArrowHead'
            : 'architectureArrowHead',
        }),
      );
      defs.append(marker);
    }
    svg.append(defs);
    this.edges = [];
    this.elements.clear();
    for (const edge of graph.edges()) {
      const path = svgElement('path', {
        class: 'architectureEdge',
        d: graph
          .edge(edge)
          .points.map(
            (point: {x: number; y: number}, index: number) =>
              (index ? 'L' : 'M') + point.x + ',' + point.y,
          )
          .join(' '),
        'marker-end': `url(#${this.marker})`,
      });
      this.edges.push({source: edge.v, target: edge.w, path});
      svg.append(path);
    }
    for (const node of topology.nodes) {
      const box = graph.node(node.id);
      const group = svgElement('g', {
        class:
          'architectureNode ' +
          (node.kind === 'tensor'
            ? 'architectureTensor'
            : node.compound
              ? 'architectureEntry'
              : node.port
                ? 'architecturePort'
                : 'architectureOperation'),
        transform: `translate(${box.x - node.width / 2},${box.y - node.height / 2})`,
        role: 'button',
        tabindex: 0,
        'data-architecture-id': node.id,
      });
      group.append(
        svgElement('rect', {
          width: node.width,
          height: node.height,
          rx: node.kind === 'tensor' ? 22 : 4,
        }),
      );
      if (node.kind === 'tensor') {
        group.append(
          svgElement(
            'text',
            {
              class: 'architectureTensorName',
              x: node.width / 2,
              y: 17,
              'text-anchor': 'middle',
            },
            this.label(node.name, 36),
          ),
        );
        group.append(
          svgElement('text', {
            class: 'architectureMetric',
            x: node.width / 2,
            y: 33,
            'text-anchor': 'middle',
          }),
        );
      } else {
        group.append(
          svgElement(
            'text',
            {
              class: 'architectureNodeName',
              x: node.width / 2,
              y: node.compound ? 21 : node.height / 2 + 4,
              'text-anchor': 'middle',
            },
            this.label(node.name, node.executable ? 26 : 28),
          ),
        );
        if (node.compound)
          group.append(
            svgElement(
              'text',
              {
                class: 'architectureNodeAction',
                x: node.width / 2,
                y: 39,
                'text-anchor': 'middle',
              },
              'View execution ↗',
            ),
          );
      }
      group.append(svgElement('title'));
      this.elements.set(node.id, {node, group, x: box.x, y: box.y});
      svg.append(group);
      if (node.executable) {
        const action = svgElement('g', {
          class: 'architectureExecute',
          transform: `translate(${box.x - node.width / 2},${box.y - node.height / 2})`,
          role: 'button',
          tabindex: 0,
          'data-architecture-execute': node.id,
          'aria-label': `View execution for ${node.name}`,
        });
        // The second line keeps the accepted module entry; small nodes expose the same action at their right edge.
        action.append(
          svgElement('rect', {
            x: node.compound ? 4 : node.width - 27,
            y: node.compound ? 27 : 4,
            width: node.compound ? node.width - 8 : 23,
            height: node.compound ? 22 : 26,
            rx: 3,
          }),
        );
        if (!node.compound)
          action.append(
            svgElement(
              'text',
              {
                class: 'architectureExecuteIcon',
                x: node.width - 15,
                y: 22,
                'text-anchor': 'middle',
              },
              '↗',
            ),
          );
        action.append(
          svgElement(
            'title',
            {},
            `View execution for ${node.namespace ? node.namespace + ' · ' : ''}${node.name}`,
          ),
        );
        svg.append(action);
      }
    }
    const viewport = document.createElement('div'),
      stage = document.createElement('div');
    viewport.className = 'architectureViewport';
    viewport.tabIndex = 0;
    viewport.setAttribute('role', 'region');
    viewport.setAttribute(
      'aria-label',
      'Architecture canvas. Scroll or drag background to pan. Use canvas controls to zoom.',
    );
    stage.className = 'architectureStage';
    stage.append(svg);
    viewport.append(stage);
    this.viewportListeners?.abort();
    this.svg = svg;
    this.viewport = viewport;
    this.host.replaceChildren(viewport, this.createControls());
    this.bindViewport(viewport);
  }
  private label(value: string, limit: number): string {
    return value.length > limit ? value.slice(0, limit - 1) + '…' : value;
  }

  private bindViewport(viewport: HTMLDivElement): void {
    this.viewportListeners = new AbortController();
    const signal = this.viewportListeners.signal;
    let drag: {
      id: number;
      x: number;
      y: number;
      left: number;
      top: number;
    } | null = null;
    viewport.addEventListener('scroll', () => this.saveViewport(), {
      passive: true,
      signal,
    });
    viewport.addEventListener(
      'pointerdown',
      (event) => {
        if (
          event.button !== 0 ||
          (event.target instanceof Element &&
            event.target.closest(
              '[data-architecture-id],[data-architecture-execute]',
            ))
        )
          return;
        drag = {
          id: event.pointerId,
          x: event.clientX,
          y: event.clientY,
          left: viewport.scrollLeft,
          top: viewport.scrollTop,
        };
        viewport.setPointerCapture(event.pointerId);
        viewport.classList.add('panning');
        event.preventDefault();
      },
      {signal},
    );
    viewport.addEventListener(
      'pointermove',
      (event) => {
        if (!drag || event.pointerId !== drag.id) return;
        viewport.scrollLeft = drag.left + drag.x - event.clientX;
        viewport.scrollTop = drag.top + drag.y - event.clientY;
      },
      {signal},
    );
    const stop = (event: PointerEvent) => {
      if (!drag || event.pointerId !== drag.id) return;
      drag = null;
      viewport.classList.remove('panning');
      if (viewport.hasPointerCapture(event.pointerId))
        viewport.releasePointerCapture(event.pointerId);
      this.saveViewport();
    };
    viewport.addEventListener('pointerup', stop, {signal});
    viewport.addEventListener('pointercancel', stop, {signal});
    viewport.addEventListener('lostpointercapture', stop, {signal});
    viewport.addEventListener(
      'keydown',
      (event) => {
        if (event.target !== viewport) return;
        const actions: Record<string, ArchitectureAction> = {
          '+': 'zoomIn',
          '=': 'zoomIn',
          '-': 'zoomOut',
          '0': 'readable',
          f: 'fit',
        };
        if (actions[event.key]) {
          event.preventDefault();
          this.control(actions[event.key]);
        }
      },
      {signal},
    );
  }

  private update(): void {
    const options = this.options!,
      selected = this.selectedId(),
      producers = new Set<string>(),
      consumers = new Set<string>();
    for (const edge of this.edges) {
      const producer = edge.target === selected,
        consumer = edge.source === selected;
      if (producer) producers.add(edge.source);
      if (consumer) consumers.add(edge.target);
      edge.path.classList.toggle('related', producer || consumer);
      edge.path.setAttribute(
        'marker-end',
        `url(#${this.marker}${producer || consumer ? '-relation' : ''})`,
      );
    }
    const rows = new Map(
      options.rows
        .filter((row) => row.layer === options.layer)
        .map((row) => [row.anchor, row]),
    );
    const maximum = Math.max(
      0,
      ...options.rows.map(
        (row) =>
          magnitude(architectureMetric(row, options.metric), options.metric) ??
          0,
      ),
    );
    for (const {node, group} of this.elements.values()) {
      group.classList.toggle('selected', node.id === selected);
      group.classList.toggle('producer', producers.has(node.id));
      group.classList.toggle('consumer', consumers.has(node.id));
      group.setAttribute('aria-pressed', String(node.id === selected));
      const relation = producers.has(node.id)
        ? 'Immediate producer of selected node'
        : consumers.has(node.id)
          ? 'Immediate consumer of selected node'
          : '';
      if (node.anchor) {
        const row = rows.get(node.anchor.id),
          value = architectureMetric(row, options.metric),
          metricStatus =
            row?.metrics[options.metric]?.status ??
            row?.status ??
            'Not captured';
        const text = `${options.metric} ${format(value)}`,
          severity = magnitude(value, options.metric),
          fraction = severity === null || !maximum ? 0 : severity / maximum;
        group.querySelector('.architectureMetric')!.textContent = text;
        group.classList.toggle('missing', value === null);
        group.style.setProperty(
          '--tensor-fill',
          value === null
            ? 'var(--architecture-bg)'
            : `color-mix(in srgb, var(--architecture-diff-strong) ${fraction * 100}%, var(--architecture-diff-low))`,
        );
        group.setAttribute(
          'aria-label',
          `Tensor ${node.name}, ${text}${value === null ? ', ' + metricStatus : ''}${relation ? ', ' + relation : ''}`,
        );
        group.querySelector('title')!.textContent =
          `${node.anchor.id} · ${node.name}\n${node.anchor.of}\n${text}\n${metricStatus}${relation ? '\n' + relation : ''}`;
      } else {
        const description = `${node.namespace ? node.namespace + ' · ' : ''}${node.name}`;
        group.setAttribute(
          'aria-label',
          `Inspect ${description}${relation ? ', ' + relation : ''}`,
        );
        group.querySelector('title')!.textContent =
          description + (relation ? '\n' + relation : '');
      }
    }
    this.updateControls();
  }
  private empty(message: string, error = false): void {
    this.viewportListeners?.abort();
    this.svg = undefined;
    this.viewport = undefined;
    this.elements.clear();
    this.edges = [];
    this.controls.clear();
    const paragraph = document.createElement('p');
    paragraph.className = 'architectureEmpty';
    paragraph.textContent = message;
    if (error) paragraph.setAttribute('role', 'alert');
    this.host.replaceChildren(paragraph);
  }
}
