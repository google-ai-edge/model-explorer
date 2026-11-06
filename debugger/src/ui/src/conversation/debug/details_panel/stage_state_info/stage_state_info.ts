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
  type Binding,
  Component,
  Directive,
  InjectionToken,
  OnChanges,
  type Type,
  ViewContainerRef,
  computed,
  inject,
  input,
  inputBinding,
  output,
  outputBinding,
  reflectComponentType,
} from '@angular/core';

import {
  ANALYSES,
  type FinalLogitsInspection,
  type GraphMetricPanel,
  type KvPanel,
  type StagePoint,
  type StateLocation,
} from '../../../../analyses/registry';
import type {ComparisonAlignment} from '../../../../server/contracts/capture';
import type {RangeSelection} from '../../../../widgets/charts/range_selection';
import type {StageAnalysisPanel} from '../../../stage_analysis';

/** Contract exposing signal input readers and output listeners to dynamic details bodies. */
interface DetailsBindings {
  readonly bodyInputs: Readonly<Record<string, () => unknown>>;
  readonly bodyOutputs: Readonly<Record<string, (value: unknown) => void>>;
}

const DETAILS_BINDINGS = new InjectionToken<DetailsBindings>('DETAILS_BINDINGS');

function isStateLocation(value: unknown): value is StateLocation {
  return (
    value !== null &&
    typeof value === 'object' &&
    'position' in value &&
    typeof value.position === 'number' &&
    'inspect' in value &&
    typeof value.inspect === 'boolean'
  );
}

/** Renders the registry's details body of the frame's panel, passing on the frame's inputs and
 *  outputs that the body declares, by name. */
@Directive({selector: '[detailsBody]'})
class DetailsBody implements OnChanges {
  readonly detailsBody = input.required<Type<unknown> | undefined>();
  private readonly container = inject(ViewContainerRef);
  private readonly frame = inject(DETAILS_BINDINGS);

  ngOnChanges(): void {
    this.container.clear();
    const body = this.detailsBody();
    if (!body) {
      return;
    }
    const mirror = reflectComponentType(body);
    if (!mirror) {
      return;
    }
    const bindings: Binding[] = [];
    for (const {templateName} of mirror.inputs) {
      const value = this.frame.bodyInputs[templateName];
      if (typeof value !== 'function') {
        console.warn(`${mirror.selector} reads ${templateName}, which the details lack`);
        continue;
      }
      bindings.push(inputBinding(templateName, value));
    }
    for (const {templateName} of mirror.outputs) {
      const listener = this.frame.bodyOutputs[templateName];
      if (typeof listener !== 'function') {
        console.warn(`${mirror.selector} emits ${templateName}, which the details lack`);
        continue;
      }
      bindings.push(outputBinding(templateName, listener));
    }
    this.container.createComponent(body, {bindings});
  }
}

/** The details panel of an analysis panel: the box every analysis' details share, holding the
 *  body the registry gives the panel's analysis. */
@Component({
  selector: 'stage-state-info',
  imports: [DetailsBody],
  providers: [{provide: DETAILS_BINDINGS, useExisting: StageStateInfo}],
  templateUrl: './stage_state_info.ng.html',
  styles: [
    `
      @use '../../../../theme/analysis_details' as details;
      @use '../../../../theme/token_diff' as token-diff;
      @include details.body;
      :host {
        @include token-diff.activation-colors;
        container: analysis-info / inline-size;
        --info-property-gap: 0px;
        --me-font-family: Arial, Helvetica, sans-serif;
        --info-property-line-height: normal;
        --info-table-action-size: 14px;
        --info-table-action-gap: 1px;
        font-family: var(--me-font-family);
        letter-spacing: normal;
        display: block;
        min-width: 0;
        font-size: 12px;
      }
    `,
  ],
})
export class StageStateInfo implements DetailsBindings {
  readonly panel = input.required<StageAnalysisPanel>();
  readonly label = input.required<string>();
  readonly range = input.required<{start: number; end: number}>();
  readonly graph = input<GraphMetricPanel>();
  readonly alignment = input<ComparisonAlignment>('content');
  readonly kv = input<KvPanel>();
  readonly finalLogits = input<FinalLogitsInspection>();
  readonly kvSelection = input<(RangeSelection & {kind: string | null}) | null>(null);
  readonly graphSelection = input<RangeSelection | null>(null);
  readonly points = input<StagePoint[]>([]);
  /** The row the panel's selection stands for (AnalysisPanels.inspectedRow), or null. */
  readonly inspectedRow = input<number | null>(null);
  readonly locateAnalysis = output<void>();
  readonly locateCache = output<'key' | 'value'>();
  readonly locateAggregation = output<void>();
  readonly locateMetric = output<void>();
  readonly locateView = output<void>();
  readonly locateLogitElement = output<number>();
  readonly locateTensorPosition = output<void>();
  readonly locateScope = output<'positions' | 'layers'>();
  readonly locateGraph = output<void>();
  readonly locateFinal = output<string>();
  readonly location = output<StateLocation>();
  /** The details body of the panel's analysis; none for a kind the registry does not know. */
  readonly body = computed<Type<unknown> | undefined>(
    () => Object.values(ANALYSES).find((a) => a.kind === this.panel().kind)?.details,
  );

  readonly bodyInputs: Readonly<Record<string, () => unknown>> = {
    panel: this.panel,
    label: this.label,
    range: this.range,
    graph: this.graph,
    alignment: this.alignment,
    kv: this.kv,
    finalLogits: this.finalLogits,
    kvSelection: this.kvSelection,
    graphSelection: this.graphSelection,
    points: this.points,
    inspectedRow: this.inspectedRow,
  };

  readonly bodyOutputs: Readonly<Record<string, (value: unknown) => void>> = {
    locateAnalysis: () => {
      this.locateAnalysis.emit();
    },
    locateCache: (value) => {
      if (value === 'key' || value === 'value') {
        this.locateCache.emit(value);
      }
    },
    locateAggregation: () => {
      this.locateAggregation.emit();
    },
    locateMetric: () => {
      this.locateMetric.emit();
    },
    locateView: () => {
      this.locateView.emit();
    },
    locateLogitElement: (value) => {
      if (typeof value === 'number') {
        this.locateLogitElement.emit(value);
      }
    },
    locateTensorPosition: () => {
      this.locateTensorPosition.emit();
    },
    locateScope: (value) => {
      if (value === 'positions' || value === 'layers') {
        this.locateScope.emit(value);
      }
    },
    locateGraph: () => {
      this.locateGraph.emit();
    },
    locateFinal: (value) => {
      if (typeof value === 'string') {
        this.locateFinal.emit(value);
      }
    },
    location: (value) => {
      if (isStateLocation(value)) {
        this.location.emit(value);
      }
    },
  };
}
