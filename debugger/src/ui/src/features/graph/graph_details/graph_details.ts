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

import {NgTemplateOutlet} from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {ExplorerInfoSection} from '../../../shared/explorer_info_section/explorer_info_section';
import {ExplorerInfoValue} from '../../../shared/explorer_info_value/explorer_info_value';
import {GraphInspectionService, GraphSide} from '../graph_inspection_service';

type Evidence = Record<string, unknown>;
interface DetailRow {
  label: string;
  value?: unknown;
  reference?: unknown;
  target?: unknown;
  metric?: string;
}
interface DetailSection {
  id: string;
  title: string;
  comparison: boolean;
  expanded: boolean;
  rows: DetailRow[];
}
function toEvidenceRecord(value: unknown): Evidence {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Evidence)
    : {};
}

function first(...values: unknown[]): unknown {
  return values.find((value) => value !== null && value !== undefined);
}

function attributes(
  value: {key: string; value: string}[] | undefined,
): Evidence {
  return Object.fromEntries(
    (value ?? []).map((attribute) => [attribute.key, attribute.value]),
  );
}

@Component({
  selector: 'graph-details',
  standalone: true,
  imports: [
    MatButtonModule,
    MatIconModule,
    NgTemplateOutlet,
    ExplorerInfoSection,
    ExplorerInfoValue,
  ],
  templateUrl: './graph_details.ng.html',
  styleUrl: './graph_details.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GraphDetails {
  readonly inspection = inject(GraphInspectionService);
  readonly state = this.inspection.state;
  readonly sections = computed<DetailSection[]>(() => {
    const details = this.inspection.details();
    const context = this.inspection.context();
    if (!details || !context) return [];
    const operation = this.inspection.operation();
    const records = [
      this.inspection.tensor('ref', this.inspection.selectedReference()),
      this.inspection.tensor('target', this.inspection.selectedTarget()),
    ];
    const pair = !operation && !!records[0] && !!records[1];
    const tensor = !operation && records.some(Boolean);
    const activeSide =
      operation?.side ??
      this.inspection.selectedTensor()?.side ??
      (records[0] ? 'ref' : 'target');
    const batch = this.state
      .session()
      ?.batches.find((batch) => batch.batch === context.batch);
    const sideIds: readonly GraphSide[] = ['ref', 'target'];
    const sides = sideIds.map((side, index) => {
      const record = operation ? undefined : records[index];
      const execution = details.executions.find((run) => run.id === side);
      const graphId =
        operation?.side === side ? operation.graphId : record?.graph;
      const graph = execution?.graphs.find((graph) => graph.id === graphId);
      const nodeId = operation?.side === side ? operation.nodeId : record?.node;
      const node = graph?.nodes.find((node) => node.id === nodeId);
      const output = record
        ? node?.outputsMetadata?.find((output) => output.id === record.output)
        : undefined;
      const run = this.state.session()?.runs.find((run) => run.id === side);
      return {
        side,
        record,
        execution,
        graph,
        node,
        output,
        run,
        recordData: toEvidenceRecord(record),
        outputAttrs: attributes(output?.attrs),
      };
    });
    type SideEvidence = (typeof sides)[number];
    const selected = sides[activeSide === 'ref' ? 0 : 1];
    const sections: DetailSection[] = [];
    const properties = (
      id: string,
      title: string,
      rows: DetailRow[],
      expanded = false,
    ) => sections.push({id, title, comparison: false, rows, expanded});
    const comparison = (
      id: string,
      title: string,
      descriptors: [string, (value: SideEvidence) => unknown][],
      expanded = false,
    ) => {
      sections.push({
        id,
        title,
        comparison: pair,
        expanded,
        rows: descriptors.map(([label, read]) =>
          pair
            ? {label, reference: read(sides[0]), target: read(sides[1])}
            : {label, value: read(selected)},
        ),
      });
    };
    const dynamic = (
      id: string,
      title: string,
      read: (value: SideEvidence) => Evidence,
      empty: string,
    ) => {
      const keys = [
        ...new Set(
          (pair ? sides : [selected]).flatMap((side) =>
            Object.keys(read(side)),
          ),
        ),
      ];
      comparison(
        id,
        title,
        keys.length
          ? keys.map((key) => [key, (side) => read(side)[key]])
          : [[empty, () => undefined]],
      );
    };

    if (tensor || operation) {
      const tensorDescriptors: [string, (side: SideEvidence) => unknown][] =
        operation
          ? []
          : [
              ['Record ID', (side) => side.record?.id],
              ['Output port', (side) => side.record?.output],
              [
                'Tensor name',
                (side) =>
                  first(
                    side.recordData['tensor'],
                    side.outputAttrs['tensor_name'],
                  ),
              ],
              [
                'Shape',
                (side) =>
                  first(
                    side.record?.shape,
                    side.outputAttrs['shape'],
                    side.outputAttrs['tensor_shape'],
                  ),
              ],
              [
                'Dtype',
                (side) =>
                  first(
                    side.record?.dtype,
                    side.outputAttrs['dtype'],
                    side.outputAttrs['tensor_dtype'],
                  ),
              ],
            ];
      comparison(
        'identity',
        operation ? 'Operation info' : 'Tensor info',
        [
          ['Run', (side) => side.execution?.id],
          ['Graph', (side) => side.graph?.id],
          ['Producer node', (side) => side.node?.id],
          ['Label', (side) => side.node?.label],
          ['Namespace', (side) => side.node?.namespace],
          ...tensorDescriptors,
        ],
        true,
      );
    } else {
      properties(
        'identity',
        'Semantic node',
        [
          {label: 'ID', value: details.node.id},
          {label: 'Label', value: details.node.label},
          {label: 'Namespace', value: details.node.namespace},
          {label: 'Anchor', value: details.anchor},
        ],
        true,
      );
    }
    if (!operation) {
      const result = this.inspection.result();
      const metricNames = [
        ...new Set([
          ...this.state.metrics,
          ...Object.keys(result?.metrics ?? {}),
        ]),
      ];
      properties(
        'numerical',
        'Numerical comparison',
        [
          {
            label: 'Status',
            value: this.inspection.comparing()
              ? 'Computing'
              : this.inspection.status(),
          },
          ...metricNames.flatMap((metric) => [
            {label: metric, value: result?.metrics[metric]?.value, metric},
            {label: metric + ' status', value: result?.metrics[metric]?.status},
          ]),
        ],
        true,
      );
    }
    properties('observation', 'Observation and capture', [
      {label: 'Capture', value: this.state.captureId()},
      {label: 'Model', value: this.state.session()?.model},
      {label: 'Layer', value: context.layer},
      {label: 'Batch', value: context.batch},
      {label: 'Turn', value: batch?.turn},
      {label: 'Phase', value: batch?.phase},
      {label: 'Index', value: batch?.index},
      {label: 'Step', value: batch?.step},
      {label: 'Signature', value: batch?.signature},
      {label: 'Batch graph', value: batch?.graph},
      {label: 'Forward IDs by run', value: batch?.forward_ids},
      {label: 'Forward ID', value: batch?.forward_id},
      {label: 'Semantic context', value: context.semantic},
      {label: 'Comparison basis', value: batch?.comparison_basis},
    ]);
    if (tensor) {
      comparison('tensor-metadata', 'Tensor metadata', [
        ['Sample', (side) => side.record?.sample],
        ['Shape', (side) => side.record?.shape],
        ['Dtype', (side) => side.record?.dtype],
        [
          'Layout',
          (side) =>
            first(side.recordData['layout'], side.outputAttrs['layout']),
        ],
        [
          'Byte size',
          (side) =>
            first(side.recordData['nbytes'], side.recordData['byte_size']),
        ],
      ]);
      dynamic(
        'output',
        'Selected output attributes',
        (side) => side.outputAttrs,
        'Output attributes',
      );
      comparison('statistics', 'Captured tensor statistics', [
        [
          'Statistics',
          (side) =>
            first(
              side.recordData['stats'],
              side.recordData['statistics'],
              toEvidenceRecord(side.recordData['data'])['stats'],
            ),
        ],
      ]);
      dynamic('record', 'Captured record', (side) => side.recordData, 'Record');
      comparison('storage', 'Tensor storage', [
        [
          'File',
          (side) => first(side.recordData['file'], side.recordData['path']),
        ],
        ['Data', (side) => side.recordData['data']],
        ['Values', (side) => side.recordData['values']],
      ]);
    }
    if (tensor || operation) {
      dynamic(
        'producer',
        'Producer attributes',
        (side) => attributes(side.node?.attrs),
        'Attributes',
      );
      comparison('ports', 'Input and output ports', [
        ['Inputs', (side) => toEvidenceRecord(side.node)['inputsMetadata']],
        ['Outputs', (side) => side.node?.outputsMetadata],
      ]);
      comparison('connections', 'Connections', [
        ['Incoming edges', (side) => side.node?.incomingEdges],
        [
          'Upstream nodes',
          (side) => side.node?.incomingEdges?.map((edge) => edge.sourceNodeId),
        ],
        [
          'Downstream nodes',
          (side) =>
            side.graph?.nodes
              .filter((node) =>
                node.incomingEdges?.some(
                  (edge) => edge.sourceNodeId === side.node?.id,
                ),
              )
              .map((node) => node.id),
        ],
        [
          'Inferred edges',
          (side) =>
            first(
              toEvidenceRecord(side.graph)['edges_inferred'],
              toEvidenceRecord(side.execution)['edges_inferred'],
            ),
        ],
      ]);
      dynamic(
        'runtime',
        'Runtime',
        (side) => ({
          ...toEvidenceRecord(side.run),
          ...Object.fromEntries(
            Object.entries(toEvidenceRecord(side.execution)).filter(
              ([key]) => key !== 'graphs',
            ),
          ),
        }),
        'Run metadata',
      );
      dynamic(
        'graph',
        'Execution graph metadata',
        (side) =>
          Object.fromEntries(
            Object.entries(toEvidenceRecord(side.graph)).filter(
              ([key]) => key !== 'nodes',
            ),
          ),
        'Graph metadata',
      );
    }
    properties(
      'parameters',
      'Semantic parameters',
      Object.keys(details.parameters ?? {}).length
        ? Object.entries(details.parameters).map(([label, value]) => ({
            label,
            value,
          }))
        : [{label: 'Parameters', value: undefined}],
    );
    properties('source', 'Semantic source evidence', [
      {label: 'Binding', value: details.binding},
      ...(details.sources.length
        ? details.sources.map((source, index) => ({
            label: 'Source ' + (index + 1),
            value: source,
          }))
        : [{label: 'Source', value: undefined}]),
    ]);
    if (!operation) {
      properties('mapping', 'Mapping', [
        {
          label: 'Status',
          value: this.inspection.mapping() ? 'Draft' : details.saved.status,
        },
        {
          label: 'Reference',
          value: this.inspection.selectedReference() || undefined,
        },
        {label: 'Target', value: this.inspection.selectedTarget() || undefined},
        {label: 'Saved selection', value: details.saved.record},
        {label: 'Scope', value: {capture: this.state.captureId(), ...context}},
      ]);
    }
    const raw: DetailRow[] = [{label: 'Observation', value: batch}];
    if (tensor || operation) {
      for (const side of pair ? sides : [selected]) {
        const label = side.side === 'ref' ? 'Reference' : 'Target';
        if (!operation) {
          raw.push(
            {label: label + ' record', value: side.record},
            {label: label + ' output', value: side.output},
          );
        }
        raw.push(
          {label: label + ' producer', value: side.node},
          {label: label + ' graph', value: side.graph},
          {label: label + ' execution', value: side.execution},
          {label: label + ' run', value: side.run},
        );
      }
    }
    raw.push(
      {label: 'Semantic node', value: details.node},
      {label: 'Semantic anchor', value: details.anchor},
      {label: 'Parameters', value: details.parameters},
      {label: 'Binding', value: details.binding},
      {label: 'Sources', value: details.sources},
    );
    if (!operation) {
      raw.push(
        {label: 'Saved mapping', value: details.saved},
        {label: 'Current comparison', value: this.inspection.result()},
        {label: 'Node response', value: details},
        {label: 'Capture session', value: this.state.session()},
      );
    }
    properties('raw', 'Original evidence', raw);
    return sections;
  });

  text(value: unknown): string {
    if (value === undefined || value === null) return 'Not captured';
    return typeof value === 'object'
      ? JSON.stringify(value, null, 2)
      : String(value);
  }
  retry() {
    const context = this.inspection.context();
    if (context) void this.inspection.select(context);
  }
}
