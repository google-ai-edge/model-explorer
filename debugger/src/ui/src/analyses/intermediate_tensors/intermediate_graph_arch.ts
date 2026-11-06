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
  Component,
  DestroyRef,
  ElementRef,
  effect,
  inject,
  input,
  output,
  signal,
  viewChild,
} from '@angular/core';
import type {ExecutionGraphCollection} from '../../server/contracts/artifacts';
import type {GraphArchModel, GraphSide} from './graph_arch_data';

export interface GraphArchContext {
  stage: string;
  metricId?: string;
  metricName?: string;
  position?: number | null;
  legend?: boolean;
  summary: 'min' | 'max';
  /** Each layer's output at the position, as the server compared it; null where it has none. */
  layers: (number | null)[];
  colors: Record<string, {bgColor: string; textColor: string}>;
  formatted: Record<string, string>;
  /** The model the Graph Diff draws: its skeleton, its anchors' tags and bindings, its points'
   *  tags, all the server's (graph_arch_data). */
  model?: GraphArchModel;
}
/** The Graph Diff's ask for one side's execution graph of a signature. */
export interface GraphRequest {
  side: GraphSide;
  signature: string;
}
/** The answer to a GraphRequest: the server's graph collection, or why there is none. */
export interface GraphDelivery extends GraphRequest {
  artifact: string;
  collection?: ExecutionGraphCollection;
  error?: string;
}
/** One side's tensor of a pair the editor asks the server to save: an operator output of the
 *  signature's graph it shows. */
export interface GraphPairTensor {
  signature: string;
  node: string;
  output: string;
}
/** A pair made by hand in the Mapping pane, for the server to save. */
export interface GraphPairSave {
  ref: GraphPairTensor;
  target: GraphPairTensor;
}
export interface GraphArchSection {
  label: string;
  tensorItems?: {label: string; children: {label: string; value: string}[]}[];
  items: {label: string; value: string; metricBg?: string; shape?: string; selected?: boolean}[];
}
export interface GraphTensorPair {
  referenceId: string;
  targetId: string;
  reference: string;
  target: string;
  referenceName: string;
  targetName: string;
  referenceSource?: string;
  targetSource?: string;
  referenceShape?: string;
  targetShape?: string;
  /** The server's id of a pair made by hand; absent for one the server makes itself. */
  manualId?: string;
  selected: boolean;
  metric?: {
    state: 'unpaired' | 'missing' | 'pending' | 'ready';
    text: string;
    label: string;
    bg?: string;
    fg?: string;
  };
}
export interface GraphMappingView {
  active: boolean;
  editing: boolean;
  canResetDefaults?: boolean;
  title: string;
  anchor: string;
  outputs: {id: string; label: string}[];
  pairs?: GraphTensorPair[];
  draft?: {
    reference: string;
    target: string;
    ready: boolean;
    conflicts: number;
    existing: boolean;
  } | null;
}

/** The Graph Diff, a same-origin page in a frame. It receives the server's values at the selected
 *  position and the model's skeleton, and asks for each execution graph it shows. */
@Component({
  selector: 'intermediate-graph-arch',
  template: `<iframe
    #frame
    title="Graph Diff architecture"
    src="graph-arch-preview/index.html?v=156"
    (load)="sync()"
  ></iframe>`,
  styles: [
    `
      :host {
        display: block;
        min-height: 480px;
        height: 100%;
      }
      iframe {
        display: block;
        width: 100%;
        height: 100%;
        border: 0;
        background: var(--chart-canvas);
      }
    `,
  ],
})
export class IntermediateGraphArch {
  readonly context = input.required<GraphArchContext>();
  readonly selection = output<GraphArchSection[]>();
  readonly graphRequest = output<GraphRequest>();
  /** A pair made by hand, to save; and pairs made by hand, by id, to delete. */
  readonly pairSave = output<GraphPairSave>();
  readonly pairDelete = output<string[]>();
  readonly mapping = signal<GraphMappingView>({
    active: false,
    editing: false,
    title: '',
    anchor: '',
    outputs: [],
  });
  private readonly frame = viewChild<ElementRef<HTMLIFrameElement>>('frame');
  private readonly channel = 'intermediate-graph-arch';
  mappingAction(
    action: 'back' | 'edit' | 'output' | 'confirm' | 'cancel' | 'reset-default',
    anchor?: string,
  ): void {
    this.frame()?.nativeElement.contentWindow?.postMessage(
      {channel: this.channel, type: 'mapping-action', action, anchor},
      location.origin,
    );
  }
  /** Hands the frame an execution graph it asked for. */
  deliverGraph(delivery: GraphDelivery): void {
    this.frame()?.nativeElement.contentWindow?.postMessage(
      {channel: this.channel, type: 'execution-graph', ...delivery},
      location.origin,
    );
  }
  resetView(): void {
    this.frame()?.nativeElement.contentWindow?.postMessage(
      {channel: this.channel, type: 'reset-view'},
      location.origin,
    );
  }
  private surfaceColor(variable: string): string {
    const frame = this.frame()?.nativeElement;
    if (!frame?.parentElement) {
      return '';
    }
    const probe = document.createElement('span');
    probe.style.cssText = `position:absolute;visibility:hidden;pointer-events:none;background:var(${variable})`;
    frame.parentElement.append(probe);
    const color = getComputedStyle(probe).backgroundColor;
    probe.remove();
    return color;
  }
  sync(): void {
    const frame = this.frame()?.nativeElement;
    if (!frame?.contentWindow) {
      return;
    }
    const styles = getComputedStyle(frame);
    frame.contentWindow.postMessage(
      {
        channel: this.channel,
        type: 'context',
        ...this.context(),
        theme: document.documentElement.dataset['theme'] ?? 'light',
        palette: {
          reference: this.surfaceColor('--chart-reference-surface'),
          target: this.surfaceColor('--chart-target-surface'),
          scrollThumb: styles.getPropertyValue('--me-outline-color').trim(),
          canvas: styles.getPropertyValue('--chart-canvas').trim(),
          list: styles.getPropertyValue('--me-surface-container-low-color').trim(),
          low: styles.getPropertyValue('--heat-low').trim(),
          high: styles.getPropertyValue('--heat-high').trim(),
        },
      },
      location.origin,
    );
  }
  constructor() {
    effect(() => {
      this.context();
      this.sync();
    });
    const receive = (event: MessageEvent): void => {
      if (
        event.origin !== location.origin ||
        event.source !== this.frame()?.nativeElement.contentWindow ||
        event.data?.channel !== this.channel
      ) {
        return;
      }
      if (event.data.type === 'ready') {
        this.mapping.set({active: false, editing: false, title: '', anchor: '', outputs: []});
        this.sync();
      }
      if (event.data.type === 'graph-request') {
        const side = event.data.side;
        const signature = event.data.signature;
        if ((side === 'ref' || side === 'target') && typeof signature === 'string') {
          this.graphRequest.emit({side, signature: signature.slice(0, 200)});
        }
        return;
      }
      if (event.data.type === 'pair-save') {
        const tensor = (value: unknown): GraphPairTensor | null => {
          if (!value || typeof value !== 'object') {
            return null;
          }
          const record = value as Record<string, unknown>;
          return typeof record['signature'] === 'string' &&
            typeof record['node'] === 'string' &&
            typeof record['output'] === 'string'
            ? {
                signature: record['signature'].slice(0, 200),
                node: record['node'].slice(0, 200),
                output: record['output'].slice(0, 200),
              }
            : null;
        };
        const ref = tensor(event.data.ref);
        const target = tensor(event.data.target);
        if (ref && target) {
          this.pairSave.emit({ref, target});
        }
        return;
      }
      if (event.data.type === 'pair-delete') {
        const ids = Array.isArray(event.data.ids)
          ? event.data.ids
              .filter((id: unknown): id is string => typeof id === 'string')
              .slice(0, 512)
          : [];
        if (ids.length) {
          this.pairDelete.emit(ids);
        }
        return;
      }
      if (event.data.type === 'mapping-view') {
        this.mapping.set({
          active: event.data.active === true,
          editing: event.data.editing === true,
          canResetDefaults: event.data.canResetDefaults === true,
          // The block's title, with the server's reason when it is its layer's operators.
          title: String(event.data.title ?? '').slice(0, 1000),
          anchor: String(event.data.anchor ?? '').slice(0, 200),
          draft: event.data.draft ?? null,
          pairs: Array.isArray(event.data.pairs)
            ? event.data.pairs.filter(
                (pair: GraphTensorPair) =>
                  typeof pair?.referenceId === 'string' && typeof pair?.targetId === 'string',
              )
            : [],
          outputs: Array.isArray(event.data.outputs)
            ? event.data.outputs
                .slice(0, 30)
                .filter(
                  (item: {id: string; label: string}) =>
                    typeof item?.id === 'string' && typeof item?.label === 'string',
                )
            : [],
        });
        return;
      }
      if (event.data.type !== 'selection' || !Array.isArray(event.data.sections)) {
        return;
      }
      const sections: GraphArchSection[] = event.data.sections
        .slice(0, 12)
        .filter(
          (section: GraphArchSection) =>
            typeof section?.label === 'string' && Array.isArray(section.items),
        )
        .map((section: GraphArchSection) => ({
          label: section.label.slice(0, 100),
          tensorItems: Array.isArray(section.tensorItems)
            ? section.tensorItems
                .filter((item) => typeof item?.label === 'string' && Array.isArray(item.children))
                .map((item) => ({
                  label: item.label.slice(0, 300),
                  children: item.children
                    .filter(
                      (child) =>
                        typeof child?.label === 'string' && typeof child.value === 'string',
                    )
                    .map((child) => ({
                      label: child.label.slice(0, 200),
                      value: child.value.slice(0, 4000),
                    })),
                }))
            : undefined,
          items: section.items
            .slice(0, 100)
            .filter((item) => typeof item?.label === 'string' && typeof item.value === 'string')
            .map((item) => ({
              label: item.label.slice(0, 200),
              value: item.value.slice(0, 4000),
              shape: typeof item.shape === 'string' ? item.shape.slice(0, 2000) : undefined,
              metricBg: typeof item.metricBg === 'string' ? item.metricBg.slice(0, 500) : undefined,
              selected: item.selected === true,
            })),
        }));
      this.selection.emit(sections);
    };
    window.addEventListener('message', receive);
    const theme = new MutationObserver(() => this.sync());
    theme.observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
    inject(DestroyRef).onDestroy(() => {
      window.removeEventListener('message', receive);
      theme.disconnect();
    });
  }
}
