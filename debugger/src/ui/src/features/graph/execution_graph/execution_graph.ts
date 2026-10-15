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
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  HostListener,
  effect,
  input,
  output,
  signal,
  viewChild,
} from '@angular/core';
import {NodeDetails} from '../../../data/contracts/types';

export interface ExecutionOperation {
  side: 'ref' | 'target';
  nodeId: string;
  graphId: string;
}

/** The iframe isolates the pinned visualizer's Angular/Zone runtime from this app. */
@Component({
  selector: 'app-execution-graph',
  standalone: true,
  template: `<iframe
    #frame
    src="graph_execution/index.html?v=e0f5fcfb8bc662c0"
    title="Reference and Target execution graphs"
    (load)="send()"
  ></iframe>`,
  styles: `
    :host {
      display: block;
      position: relative;
      width: 100%;
      height: 100%;
      min-width: 0;
      min-height: 160px;
    }
    iframe {
      display: block;
      border: 0;
      width: 100%;
      height: 100%;
      background: var(--graph-canvas);
    }
  `,
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ExecutionGraph {
  readonly details = input<NodeDetails | null>(null);
  readonly reference = input('');
  readonly target = input('');
  readonly metric = input('mse');
  readonly metrics = input<Record<string, number | null>>({});
  readonly mapping = input(false);
  readonly dark = input(false);
  readonly operation = input<ExecutionOperation | null>(null);
  readonly tensorSelected = output<{side: 'ref' | 'target'; id: string}>();
  readonly operationSelected = output<ExecutionOperation>();
  private readonly frame = viewChild<ElementRef<HTMLIFrameElement>>('frame');
  private readonly ready = signal(false);

  constructor() {
    effect(() => {
      this.ready();
      this.send();
    });
  }

  send() {
    const details = this.details();
    const payload = {
      // The frame renders topology and records only; sources and saved mappings stay here.
      details: details ? {tensors: details.tensors, executions: details.executions} : null,
      reference: this.reference(),
      target: this.target(),
      metric: this.metric(),
      metrics: this.metrics(),
      mapping: this.mapping(),
      operation: this.operation(),
      theme: this.dark() ? 'dark' : 'light',
    };
    this.post({type: 'render', payload});
  }

  fit() {
    this.post({type: 'action', action: 'fit'});
  }
  readable() {
    this.post({type: 'action', action: 'readable'});
  }

  private post(message: Record<string, unknown>) {
    this.frame()?.nativeElement.contentWindow?.postMessage(
      {channel: 'model-debugger-execution', ...message},
      location.origin,
    );
  }

  @HostListener('window:message', ['$event'])
  receive(event: MessageEvent) {
    if (
      event.origin !== location.origin ||
      event.source !== this.frame()?.nativeElement.contentWindow
    )
      return;
    const message = event.data;
    if (!message || message.channel !== 'model-debugger-execution') return;
    if (message.type === 'ready') {
      this.ready.set(true);
      this.send();
      return;
    }
    const hit = message.hit;
    if (!hit || (hit.side !== 'ref' && hit.side !== 'target')) return;
    const details = this.details();
    if (
      message.type === 'tensorSelected' &&
      typeof hit.id === 'string' &&
      details?.tensors.some((record) => record.id === hit.id)
    ) {
      this.tensorSelected.emit({side: hit.side, id: hit.id});
    } else if (
      message.type === 'operationSelected' &&
      !this.mapping() &&
      typeof hit.nodeId === 'string' &&
      typeof hit.graphId === 'string' &&
      details?.executions.some((run) =>
        run.graphs.some(
          (graph) => graph.id === hit.graphId && graph.nodes.some((node) => node.id === hit.nodeId),
        ),
      )
    ) {
      this.operationSelected.emit({side: hit.side, nodeId: hit.nodeId, graphId: hit.graphId});
    }
  }
}
