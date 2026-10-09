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
  computed,
  DestroyRef,
  effect,
  inject,
  Injectable,
  signal,
  untracked,
} from '@angular/core';
import type {
  ComparisonRow,
  NodeDetails,
  Selection,
  SelectionResult,
  TensorRecord,
} from '../../data/contracts/types';
import {ReportApiService} from '../../data/report_api_service';
import {ReportStateService} from '../../data/report_state_service';

export interface GraphInspectionContext {
  layer: number;
  batch: number;
  semantic: string;
}
export type GraphSide = 'ref' | 'target';
export interface GraphTensorSelection {
  side: GraphSide;
  id: string;
}
export interface GraphOperationSelection {
  side: GraphSide;
  nodeId: string;
  graphId: string;
}
interface PairSnapshot {
  reference: string;
  target: string;
  result: SelectionResult | null;
}

/** Selection and draft state are scoped to the current capture observation. */
@Injectable() // Provided by GraphWorkspace so it is destroyed with the view.
export class GraphInspectionService {
  readonly state = inject(ReportStateService);
  private readonly api = inject(ReportApiService);
  readonly context = signal<GraphInspectionContext | null>(null);
  readonly details = signal<NodeDetails | null>(null);
  readonly error = signal('');
  readonly loading = signal(false);
  readonly selectedReference = signal('');
  readonly selectedTarget = signal('');
  readonly selectedTensor = signal<GraphTensorSelection | null>(null);
  readonly result = signal<SelectionResult | null>(null);
  readonly mapping = signal(false);
  readonly operation = signal<GraphOperationSelection | null>(null);
  readonly comparing = signal(false);
  readonly saving = signal(false);
  readonly comparisonError = signal('');
  readonly message = signal('');
  readonly status = computed(() => {
    if (this.operation()) return 'Operation';
    if (this.result()) return this.result()!.status;
    if (this.comparisonError()) return 'Comparison unavailable';
    if (this.mapping())
      return this.selectedReference() && this.selectedTarget()
        ? 'Awaiting comparison'
        : 'Select two captured tensors';
    if (
      this.selectedTensor() &&
      (!this.selectedReference() || !this.selectedTarget())
    )
      return 'Unmapped tensor';
    return (
      (this.details()?.saved.status.startsWith('stale')
        ? this.details()!.saved.status
        : this.row()?.status) ?? 'Not captured'
    );
  });
  readonly metrics = computed(() =>
    this.operation() ? {} : (this.result()?.metrics ?? {}),
  );
  readonly scalarMetrics = computed<Record<string, number | null>>(() =>
    Object.fromEntries(
      Object.entries(this.metrics()).map(([key, metric]) => [
        key,
        metric.value ?? null,
      ]),
    ),
  );
  readonly canMap = computed(
    () =>
      !this.operation() &&
      !this.loading() &&
      !this.saving() &&
      !!this.selectedTensor() &&
      !!this.tensor(this.selectedTensor()!.side, this.selectedTensor()!.id),
  );
  readonly canSave = computed(
    () =>
      this.mapping() &&
      !this.comparing() &&
      !this.saving() &&
      this.result()?.status === 'ok' &&
      this.result()!.reference === this.selectedReference() &&
      this.result()!.target === this.selectedTarget(),
  );
  private captureId: string | null = null;
  private generation = 0;
  private comparisonGeneration = 0;
  private nodeController?: AbortController;
  private comparisonController?: AbortController;
  private committed: PairSnapshot = {reference: '', target: '', result: null};
  private awaitingRows = false;

  constructor() {
    effect(() => {
      const captureId = this.state.captureId(),
        batch = this.state.batchId(),
        layer = this.state.layer();
      untracked(() => {
        const context = this.context();
        if (
          context &&
          (captureId !== this.captureId ||
            context.batch !== batch ||
            context.layer !== layer)
        )
          this.close();
      });
    });
    effect(() => {
      const comparison = this.state.comparison();
      untracked(() => {
        if (
          this.awaitingRows &&
          comparison?.batch === this.context()?.batch &&
          this.details() &&
          !this.mapping()
        ) {
          this.awaitingRows = false;
          this.initialize(this.details()!);
        }
      });
    });
    inject(DestroyRef).onDestroy(() => this.close());
  }

  tensor(side: GraphSide, id: string): TensorRecord | undefined {
    return this.details()?.tensors.find(
      (tensor) => tensor.run === side && tensor.id === id,
    );
  }

  async select(context: GraphInspectionContext) {
    this.close();
    this.captureId = this.state.captureId();
    this.context.set({...context});
    const generation = this.generation,
      captureId = this.captureId;
    const controller = new AbortController();
    this.nodeController = controller;
    this.loading.set(true);
    try {
      const details = await this.api.node(
        captureId,
        context.layer,
        context.batch,
        context.semantic,
        controller.signal,
      );
      if (!this.current(generation) || controller.signal.aborted) return;
      this.details.set(details);
      this.initialize(details);
    } catch (error) {
      if (this.current(generation) && !controller.signal.aborted)
        this.error.set(String(error));
    } finally {
      if (this.current(generation) && !controller.signal.aborted)
        this.loading.set(false);
    }
  }

  private current(generation: number) {
    const context = this.context();
    return (
      generation === this.generation &&
      !!context &&
      this.captureId === this.state.captureId() &&
      context.batch === this.state.batchId() &&
      context.layer === this.state.layer()
    );
  }

  private row(): ComparisonRow | undefined {
    const context = this.context(),
      comparison = this.state.comparison();
    if (
      !context ||
      comparison?.batch !== context.batch ||
      !context.semantic.startsWith('anchor:')
    )
      return;
    return comparison.rows.find(
      (row) =>
        row.layer === context.layer && row.anchor === context.semantic.slice(7),
    );
  }

  private initialize(details: NodeDetails) {
    const row = this.row(),
      saved = details.saved;
    const record = saved.status === 'saved' ? saved.record : undefined;
    const reference = record?.reference ?? row?.reference ?? '';
    const target = record?.target ?? row?.target ?? '';
    const validReference = this.tensor('ref', reference) ? reference : '';
    const validTarget = this.tensor('target', target) ? target : '';
    const result =
      !saved.status.startsWith('stale') &&
      row?.reference === validReference &&
      row.target === validTarget &&
      row.shape != null &&
      validReference &&
      validTarget
        ? ({
            status: row.status,
            reference: validReference,
            target: validTarget,
            shape: row.shape,
            metrics: row.metrics,
          } as SelectionResult)
        : null;
    this.committed = {
      reference: validReference,
      target: validTarget,
      result: saved.comparison ?? result,
    };
    this.restoreCommitted();
    this.awaitingRows =
      !record && this.state.comparison()?.batch !== this.context()?.batch;
    if (record && !this.result() && validReference && validTarget)
      void this.compare(true);
  }

  selectTensor(selection: GraphTensorSelection) {
    if (this.saving() || !this.tensor(selection.side, selection.id)) return;
    this.awaitingRows = false;
    this.abortComparison();
    this.operation.set(null);
    this.selectedTensor.set(selection);
    if (this.mapping()) {
      (selection.side === 'ref'
        ? this.selectedReference
        : this.selectedTarget
      ).set(selection.id);
      void this.compare();
    } else if (
      selection.id ===
      (selection.side === 'ref'
        ? this.committed.reference
        : this.committed.target)
    ) {
      this.selectedReference.set(this.committed.reference);
      this.selectedTarget.set(this.committed.target);
      this.result.set(this.committed.result);
    } else {
      this.selectedReference.set(selection.side === 'ref' ? selection.id : '');
      this.selectedTarget.set(selection.side === 'target' ? selection.id : '');
    }
  }

  selectOperation(selection: GraphOperationSelection) {
    if (this.mapping() || this.saving()) return;
    const node = this.details()
      ?.executions.find((run) => run.id === selection.side)
      ?.graphs.find((graph) => graph.id === selection.graphId)
      ?.nodes.find((node) => node.id === selection.nodeId);
    if (!node) return;
    this.awaitingRows = false;
    this.abortComparison();
    this.selectedTensor.set(null);
    this.operation.set(selection);
  }

  beginMapping() {
    if (!this.canMap() || this.mapping()) return;
    this.awaitingRows = false;
    this.mapping.set(true);
    this.message.set('Select a captured output tensor on each side.');
    // Existing comparison remains valid until an endpoint changes.
  }

  cancelMapping() {
    if (!this.mapping() || this.saving()) return;
    this.abortComparison();
    this.mapping.set(false);
    this.restoreCommitted();
    this.message.set('');
    if (
      this.committed.reference &&
      this.committed.target &&
      !this.committed.result &&
      !this.details()?.saved.status.startsWith('stale')
    )
      void this.compare(true);
  }

  private restoreCommitted() {
    this.selectedReference.set(this.committed.reference);
    this.selectedTarget.set(this.committed.target);
    this.result.set(this.committed.result);
    this.operation.set(null);
    this.selectedTensor.set(
      this.committed.reference
        ? {side: 'ref', id: this.committed.reference}
        : this.committed.target
          ? {side: 'target', id: this.committed.target}
          : null,
    );
  }

  private selection(): Selection | null {
    const context = this.context(),
      reference = this.selectedReference(),
      target = this.selectedTarget();
    return context && reference && target
      ? {...context, reference, target}
      : null;
  }

  private abortComparison() {
    this.comparisonGeneration++;
    this.comparisonController?.abort();
    this.comparisonController = undefined;
    this.result.set(null);
    this.comparing.set(false);
    this.comparisonError.set('');
  }

  private async compare(commit = false) {
    this.abortComparison();
    const selection = this.selection();
    if (!selection) return;
    const generation = this.generation,
      request = this.comparisonGeneration;
    const controller = new AbortController();
    this.comparisonController = controller;
    this.comparing.set(true);
    const current = () =>
      this.current(generation) &&
      request === this.comparisonGeneration &&
      !controller.signal.aborted;
    try {
      const result = await this.api.compareSelection(
        this.captureId,
        selection,
        controller.signal,
      );
      if (!current()) return;
      this.result.set(result);
      if (commit) this.committed = {...this.committed, result};
    } catch (error) {
      if (current()) this.comparisonError.set(String(error));
    } finally {
      if (current()) this.comparing.set(false);
    }
  }

  async saveMapping() {
    const selection = this.selection();
    if (!this.canSave() || !selection) return;
    const generation = this.generation,
      captureId = this.captureId;
    this.saving.set(true);
    this.message.set('');
    try {
      const saved = await this.api.saveMapping(captureId, selection);
      if (this.current(generation)) {
        this.details.update((details) =>
          details ? {...details, saved} : details,
        );
        this.committed = {
          reference: selection.reference,
          target: selection.target,
          result: saved.comparison ?? this.result(),
        };
        this.mapping.set(false);
        this.restoreCommitted();
        this.message.set('Mapping saved for this layer and batch.');
      }
      if (captureId === this.state.captureId()) this.state.refreshMetrics();
    } catch (error) {
      if (this.current(generation)) this.message.set(String(error));
    } finally {
      if (this.current(generation)) this.saving.set(false);
    }
  }

  async removeMapping() {
    const context = this.context();
    if (!context || this.saving() || this.mapping()) return;
    const generation = this.generation,
      captureId = this.captureId;
    this.saving.set(true);
    this.abortComparison();
    try {
      const saved = await this.api.removeMapping(captureId, {
        ...context,
        reference: '',
        target: '',
      });
      if (this.current(generation)) {
        this.details.update((details) =>
          details ? {...details, saved} : details,
        );
        this.committed = {reference: '', target: '', result: null};
        this.restoreCommitted();
        this.awaitingRows = true;
        this.message.set('Saved mapping removed.');
      }
      if (captureId === this.state.captureId()) this.state.refreshMetrics();
    } catch (error) {
      if (this.current(generation)) {
        this.restoreCommitted();
        this.message.set(String(error));
      }
    } finally {
      if (this.current(generation)) this.saving.set(false);
    }
  }

  close() {
    this.generation++;
    this.nodeController?.abort();
    this.nodeController = undefined;
    this.abortComparison();
    this.context.set(null);
    this.details.set(null);
    this.error.set('');
    this.loading.set(false);
    this.mapping.set(false);
    this.operation.set(null);
    this.selectedTensor.set(null);
    this.selectedReference.set('');
    this.selectedTarget.set('');
    this.saving.set(false);
    this.message.set('');
    this.awaitingRows = false;
    this.committed = {reference: '', target: '', result: null};
  }
}
