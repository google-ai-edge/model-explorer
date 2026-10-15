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

export interface GraphArchContext {
  stage: string;
  metricId?: string;
  position?: number | null;
  legend?: boolean;
  simulatedMetrics?: {value: number; text: string; bgColor: string; textColor: string}[];
  summary: 'min' | 'max';
  layers: (number | null)[];
  /** Explicit semantic scope/output bindings; absent internal metrics stay empty. */
  tensorMetrics?: Record<string, Record<string, number | null>>;
  colors: Record<string, {bgColor: string; textColor: string}>;
  formatted: Record<string, string>;
}
export interface GraphArchSection {
  label: string;
  tensorItems?: {label: string; children: {label: string; value: string}[]}[];
  items: {label: string; value: string; metricBg?: string; shape?: string; selected?: boolean}[];
}
export interface GraphTensorPair {
  referenceId: string; targetId: string;
  reference: string; target: string;
  referenceName: string; targetName: string;
  referenceSource?: string; targetSource?: string;
  referenceShape?: string; targetShape?: string;
  selected: boolean;
  metric?: {state: 'unpaired' | 'missing' | 'pending' | 'ready'; text: string; label: string; bg?: string; fg?: string};
}
export interface GraphMappingView {
  active: boolean;
  editing: boolean;
  canResetDefaults?: boolean;
  title: string;
  anchor: string;
  outputs: {id: string; label: string}[];
  pairs?: GraphTensorPair[];
  draft?: {reference: string; target: string; ready: boolean; conflicts: number; existing: boolean} | null;
}

/** Same-origin, packaged design preview. It receives panel aggregates and explicitly marked simulated internal metrics. */
@Component({
  selector: 'intermediate-graph-arch',
  template: `<iframe
    #frame
    title="Graph Diff architecture"
    src="graph_arch_preview/index.html?v=150"
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
        background: var(--kv-canvas);
      }
    `,
  ],
})
export class IntermediateGraphArch {
  readonly context = input.required<GraphArchContext>();
  readonly selection = output<GraphArchSection[]>();
  readonly mapping = signal<GraphMappingView>({active: false, editing: false, title: '', anchor: '', outputs: []});
  private readonly frame = viewChild<ElementRef<HTMLIFrameElement>>('frame');
  private readonly channel = 'intermediate-graph-arch';
  mappingAction(action: 'back' | 'edit' | 'output' | 'confirm' | 'cancel' | 'reset-default', anchor?: string) {
    this.frame()?.nativeElement.contentWindow?.postMessage(
      {channel: this.channel, type: 'mapping-action', action, anchor}, location.origin,
    );
  }
  resetView() {
    this.frame()?.nativeElement.contentWindow?.postMessage(
      {channel: this.channel, type: 'reset-view'},
      location.origin,
    );
  }
  private surfaceColor(variable: string) {
    const frame = this.frame()?.nativeElement;
    if (!frame?.parentElement) return '';
    const probe = document.createElement('span');
    probe.style.cssText = `position:absolute;visibility:hidden;pointer-events:none;background:var(${variable})`;
    frame.parentElement.append(probe);
    const color = getComputedStyle(probe).backgroundColor;
    probe.remove();
    return color;
  }
  sync() {
    this.frame()?.nativeElement.contentWindow?.postMessage(
      {
        channel: this.channel,
        type: 'context',
        ...this.context(),
        theme: document.documentElement.dataset['theme'] ?? 'light',
        palette: {
          reference: this.surfaceColor('--reference-surface'),
          target: this.surfaceColor('--target-debug-surface'),
          scrollThumb: getComputedStyle(this.frame()!.nativeElement)
            .getPropertyValue('--me-outline-color')
            .trim(),
          canvas: getComputedStyle(this.frame()!.nativeElement)
            .getPropertyValue('--kv-canvas')
            .trim(),
          list: getComputedStyle(this.frame()!.nativeElement)
            .getPropertyValue('--me-surface-container-low-color')
            .trim(),
          low: getComputedStyle(this.frame()!.nativeElement).getPropertyValue('--heat-low').trim(),
          high: getComputedStyle(this.frame()!.nativeElement)
            .getPropertyValue('--heat-high')
            .trim(),
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
    const receive = (event: MessageEvent) => {
      if (
        event.origin !== location.origin ||
        event.source !== this.frame()?.nativeElement.contentWindow ||
        event.data?.channel !== this.channel
      )
        return;
      if (event.data.type === 'ready') {
        this.mapping.set({active: false, editing: false, title: '', anchor: '', outputs: []});
        this.sync();
      }
      if (event.data.type === 'mapping-view') {
        this.mapping.set({
          active: event.data.active === true,
          editing: event.data.editing === true,
          canResetDefaults: event.data.canResetDefaults === true,
          title: String(event.data.title ?? '').slice(0, 200),
          anchor: String(event.data.anchor ?? '').slice(0, 200),
          draft: event.data.draft ?? null,
          pairs: Array.isArray(event.data.pairs) ? event.data.pairs.filter(
            (pair: GraphTensorPair) => typeof pair?.referenceId === 'string' && typeof pair?.targetId === 'string'
          ) : [],
          outputs: Array.isArray(event.data.outputs) ? event.data.outputs.slice(0, 30)
            .filter((item: {id: string; label: string}) => typeof item?.id === 'string' && typeof item?.label === 'string') : [],
        });
        return;
      }
      if (event.data.type !== 'selection' || !Array.isArray(event.data.sections)) return;
      const sections: GraphArchSection[] = event.data.sections
        .slice(0, 12)
        .filter(
          (section: GraphArchSection) =>
            typeof section?.label === 'string' && Array.isArray(section.items),
        )
        .map((section: GraphArchSection) => ({
          label: section.label.slice(0, 100),
          tensorItems: Array.isArray(section.tensorItems) ? section.tensorItems
            .filter(item => typeof item?.label === 'string' && Array.isArray(item.children))
            .map(item => ({label: item.label.slice(0, 300), children: item.children
              .filter(child => typeof child?.label === 'string' && typeof child.value === 'string')
              .map(child => ({label: child.label.slice(0, 200), value: child.value.slice(0, 4000)}))})) : undefined,
          items: section.items
            .slice(0, 100)
            .filter((item) => typeof item?.label === 'string' && typeof item.value === 'string')
            .map((item) => ({label: item.label.slice(0, 200), value: item.value.slice(0, 4000),
              shape: typeof item.shape === 'string' ? item.shape.slice(0, 2000) : undefined,
              metricBg: typeof item.metricBg === 'string' ? item.metricBg.slice(0, 500) : undefined,
              selected: item.selected === true})),
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
