/**
 * @license
 * Copyright 2024 The Model Explorer Authors. All Rights Reserved.
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

import * as d3 from 'd3';
import {AppService} from './app_service';
import {
  DEFAULT_NODE_ATTRS_TABLE_FONT_SIZE,
  NODE_ANIMATION_DURATION,
  NODE_ATTRS_TABLE_FONT_SIZE_TO_HEIGHT_RATIO,
} from './common/consts';
import {ModelNode} from './common/model_graph';
import {
  RenderElement,
  RenderElementNode,
  RenderElementType,
  WebglColor,
} from './common/types';
import {
  getNodeLabelHeight,
  getNodeLabelLineHeight,
  isGroupNode,
  isOpNode,
} from './common/utils';
import {ColorVariable} from './visualizer_theme_service';
import {WebglRenderer} from './webgl_renderer';
import {WebglRendererThreejsService} from './webgl_renderer_threejs_service';

import {Injectable} from '@angular/core';

/** Service for rendering svg texts. */
@Injectable()
export class SvgTextRendererService {
  private webglRenderer!: WebglRenderer;
  private webglRendererThreejsService!: WebglRendererThreejsService;
  private textVisible = true;
  private readonly nodeIdToCustomColor = new Map<string, string>();

  constructor(private readonly appService: AppService) {}

  init(webglRenderer: WebglRenderer) {
    this.webglRenderer = webglRenderer;
    this.webglRendererThreejsService =
      webglRenderer.webglRendererThreejsService;
  }

  render(zoomTargetElement: HTMLElement | SVGElement, clearAllFirst = false) {
    const g = d3.select(zoomTargetElement);

    if (clearAllFirst) {
      g.selectAll('g.element').remove();
    }

    const nodeElementsToRender: ModelNode[] = (
      this.webglRenderer.elementsToRender.filter(
        (d) => d.type === RenderElementType.NODE,
      ) as RenderElementNode[]
    ).map((ele) => this.nodeElement(ele));

    // Bind data.
    const element = g
      .selectAll<SVGGElement, ModelNode>('g.element')
      .data(nodeElementsToRender, (d) => d.id ?? '');

    // Enter phase.
    const elementEnter = element.enter().append('g').attr('class', 'element');
    const nodeEnter = elementEnter
      .append('g')
      .classed('node', true)
      .classed('group-node', (d) => isGroupNode(d))
      .attr('data-id', (d) => this.normalizeId(d.id))
      .attr(
        'transform',
        (d) =>
          `translate(${this.webglRenderer.getNodeX(d, true)}, ${
            this.webglRenderer.getNodeY(d, true) -
            (d.nsParentId == null ? 0 : 0)
          })`,
      );

    // Node label.
    //
    // Enter phase.
    const nodeLabelEnter = nodeEnter
      .append('text')
      .classed('node-label', true)
      .attr('x', (d) => this.getNodeLabelX(d))
      .attr('y', (d) => this.getNodeLabelY(d))
      .style('fill', (d) => this.getNodeLabelColor(d))
      .style('font-size', (d) => this.getNodeLabelFontSize(d))
      .style('font-weight', (d) => (isOpNode(d) ? 400 : 500));
    // Render multi-line text labels using tspans.
    nodeLabelEnter.each((d, i, nodes) => {
      const textElement = d3.select(nodes[i]);
      const lines = this.webglRenderer.getNodeLabel(d).split('\n');
      const nodeLabelX = this.getNodeLabelX(d);
      const lineHeight = this.getNodeLabelLineHeight(d);

      textElement.selectAll('tspan').remove();
      lines.forEach((line, lineIndex) => {
        textElement
          .append('tspan')
          .attr('x', nodeLabelX)
          .attr('dy', lineIndex === 0 ? 0 : lineHeight)
          .text(line);
      });
    });

    // Update phase.
    const elementUpdate = element.merge(elementEnter);
    const nodeUpdate = elementUpdate.select('g.node');
    const elementUpdateTransition = elementUpdate
      .transition()
      .ease(d3.easeSinOut)
      .duration(this.appService.testMode ? 0 : NODE_ANIMATION_DURATION);
    const nodeUpdateTransition = elementUpdateTransition.select('g.node');

    // Update text properties
    const nodeLabelUpdate = nodeUpdate.select('text.node-label');
    nodeLabelUpdate
      .attr('x', (d) => this.getNodeLabelX(d))
      .attr('y', (d) => this.getNodeLabelY(d))
      .style('fill', (d) => this.getNodeLabelColor(d))
      .style('font-size', (d) => this.getNodeLabelFontSize(d));

    // Update tspans
    nodeLabelUpdate.each((d, i, nodes) => {
      const textElement = d3.select(nodes[i]);
      const lines = this.webglRenderer.getNodeLabel(d).split('\n');
      const nodeLabelX = this.getNodeLabelX(d);
      const lineHeight = this.getNodeLabelLineHeight(d);
      const tspans = textElement.selectAll('tspan').data(lines);

      tspans
        .enter()
        .append('tspan')
        .attr('x', nodeLabelX)
        .attr('dy', (line, lineIndex) => (lineIndex === 0 ? 0 : lineHeight))
        .text((line) => line);
      tspans
        .attr('x', nodeLabelX)
        .attr('dy', (line, lineIndex) => (lineIndex === 0 ? 0 : lineHeight))
        .text((line) => line);
      tspans.exit().remove();
    });

    // Update node positions.
    nodeUpdateTransition.attr(
      'transform',
      (d) =>
        `translate(${this.webglRenderer.getNodeX(d, true)}, ${this.webglRenderer.getNodeY(
          d,
          true,
        )})`,
    );

    // Attrs table.
    g.selectAll('g.node-attrs-table').remove();
    const fontSize =
      this.webglRenderer.appService.config()?.nodeAttrsTableFontSize ??
      DEFAULT_NODE_ATTRS_TABLE_FONT_SIZE;
    const rowHeight = fontSize * NODE_ATTRS_TABLE_FONT_SIZE_TO_HEIGHT_RATIO;
    const scale =
      fontSize /
      this.webglRenderer.webglRendererAttrsTableService.getWebglTextsFontSize();
    const attrsTableFontSize = this.getAttrsTableFontSize();
    const attrTableBgColor = this.webglRenderer.visualizerThemeService.getColor(
      ColorVariable.SURFACE_CONTAINER_LOWEST_COLOR,
    );
    for (const node of nodeElementsToRender) {
      const {rows} =
        this.webglRenderer.webglRendererAttrsTableService.generateAttrsTableDataForNode(
          node,
          0 /* index. Doesn't matter here. */,
          fontSize,
          scale,
          rowHeight,
        );

      const nodeAttrsTable = g
        .select(`g.node[data-id="${this.normalizeId(node.id)}"]`)
        .append('g')
        .classed('node-attrs-table', true);

      // Op node bg rectangle.
      if (rows.length > 0 && isOpNode(node)) {
        const padding = 16;
        const width =
          this.webglRenderer.getNodeWidth(node, true) -
          this.webglRendererThreejsService.convertXFromSceneToScreen(-padding);
        nodeAttrsTable
          .append('rect')
          .classed('node-attrs-table-bg', true)
          .attr('width', width)
          .attr(
            'height',
            this.webglRendererThreejsService.convertZFromSceneToScreen(
              rows[rows.length - 1].keyLabelData.z -
                rows[0].keyLabelData.z +
                rowHeight,
            ),
          )
          .attr('rx', 2)
          .attr(
            'x',
            this.webglRendererThreejsService.convertXFromSceneToScreen(
              -padding / 2,
            ),
          )
          .attr(
            'y',
            this.webglRendererThreejsService.convertZFromSceneToScreen(
              rows[0].keyLabelData.z - rowHeight / 2,
            ) - this.webglRenderer.getNodeY(node, true),
          )
          .style('fill', attrTableBgColor);
      }

      // Table rows.
      for (const row of rows) {
        const textElement = nodeAttrsTable
          .append('text')
          .style('font-size', attrsTableFontSize);

        // Key Label
        const keyLabel = row.keyLabelData;
        if (keyLabel) {
          const keyX =
            this.webglRendererThreejsService.convertXFromSceneToScreen(
              -keyLabel.x,
            ) - this.webglRenderer.getNodeX(node, true);
          const keyY =
            this.webglRendererThreejsService.convertZFromSceneToScreen(
              keyLabel.z,
            ) - this.webglRenderer.getNodeY(node, true);
          textElement
            .append('tspan')
            .classed('key-label', true)
            .attr('x', keyX)
            .attr('y', keyY)
            .style('fill', this.getRgbaColor(keyLabel.color))
            .text(keyLabel.label);
        }

        // Value Label
        const valueLabel = row.valueLabelData;
        if (valueLabel) {
          const valueX =
            this.webglRendererThreejsService.convertXFromSceneToScreen(
              -valueLabel.x,
            ) - this.webglRenderer.getNodeX(node, true);
          const valueY =
            this.webglRendererThreejsService.convertZFromSceneToScreen(
              valueLabel.z,
            ) - this.webglRenderer.getNodeY(node, true);
          textElement
            .append('tspan')
            .classed('key-value', true)
            .attr('x', valueX)
            .attr('y', valueY)
            .style('fill', this.getRgbaColor(valueLabel.color))
            .text(valueLabel.label);
        }
      }
    }

    // Remove any exiting elements.
    element.exit<ModelNode>().remove();
  }

  setTextsVisible(
    zoomTargetElement: HTMLElement | SVGElement,
    visible: boolean,
  ) {
    if (this.textVisible === visible) {
      return;
    }
    this.textVisible = visible;
    d3.select(zoomTargetElement).style('display', visible ? 'block' : 'none');
  }

  setTextsOpacity(
    zoomTargetElement: HTMLElement | SVGElement,
    nodeIds: string[],
    opacity: number,
  ) {
    if (nodeIds.length === 0) {
      return;
    }

    const selectors = nodeIds
      .map((nodeId) => `g.node[data-id="${this.normalizeId(nodeId)}"]`)
      .join(', ');
    d3.select(zoomTargetElement)
      .selectAll<SVGElement, ModelNode>(selectors)
      .style('opacity', opacity);
  }

  restoreTextsOpacity(zoomTargetElement: HTMLElement | SVGElement) {
    d3.select(zoomTargetElement)
      .selectAll<SVGElement, ModelNode>('g.node')
      .style('opacity', 1);
  }

  updateColorInNode(
    zoomTargetElement: HTMLElement | SVGElement,
    nodeIds: string[],
    color: string,
  ) {
    if (nodeIds.length === 0) {
      return;
    }

    for (const nodeId of nodeIds) {
      this.nodeIdToCustomColor.set(nodeId, color);
    }

    const selectors = nodeIds
      .map((nodeId) => `g.node[data-id="${this.normalizeId(nodeId)}"]`)
      .join(', ');
    d3.select(zoomTargetElement)
      .selectAll<SVGElement, ModelNode>(selectors)
      .select('text.node-label')
      .style('fill', color);
  }

  restoreColors(zoomTargetElement: HTMLElement | SVGElement) {
    if (this.nodeIdToCustomColor.size === 0) {
      return;
    }

    const nodeIds = Array.from(this.nodeIdToCustomColor.keys());
    this.nodeIdToCustomColor.clear();
    const selectors = nodeIds
      .map((nodeId) => `g.node[data-id="${this.normalizeId(nodeId)}"]`)
      .join(', ');
    d3.select(zoomTargetElement)
      .selectAll<SVGElement, ModelNode>(selectors)
      .select('text.node-label')
      .style('fill', (d) => this.getNodeLabelColor(d));
  }

  private getNodeLabelX(node: ModelNode): number {
    return this.webglRenderer.getNodeWidth(node, true) / 2;
  }

  private getNodeLabelY(node: ModelNode): number {
    return this.webglRenderer.getNodeLabelRelativeY(node, true);
  }

  private getNodeLabelColor(node: ModelNode): string {
    if (this.nodeIdToCustomColor.has(node.id)) {
      return this.nodeIdToCustomColor.get(node.id)!;
    }
    const color = this.webglRenderer.getNodeLabelColor(node);
    return this.getRgbaColor(color);
  }

  private getNodeLabelFontSize(node: ModelNode): number {
    const labelHeight = getNodeLabelHeight(node, this.appService.config());
    return Math.abs(
      this.webglRendererThreejsService.convertXFromSceneToScreen(labelHeight),
    );
  }

  private getNodeLabelLineHeight(node: ModelNode): number {
    const lineHeight = getNodeLabelLineHeight(node, this.appService.config());
    return Math.abs(
      this.webglRendererThreejsService.convertXFromSceneToScreen(lineHeight),
    );
  }

  private getAttrsTableFontSize(): number {
    const fontSize =
      this.webglRenderer.appService.config()?.nodeAttrsTableFontSize ??
      DEFAULT_NODE_ATTRS_TABLE_FONT_SIZE;
    return Math.abs(
      this.webglRendererThreejsService.convertXFromSceneToScreen(fontSize),
    );
  }

  private getRgbaColor(webglColor: WebglColor | undefined): string {
    return `rgba(${(webglColor?.r ?? 0) * 255}, ${(webglColor?.g ?? 0) * 255}, ${(webglColor?.b ?? 0) * 255}, 1)`;
  }

  private nodeElement(renderElement: RenderElement): ModelNode {
    return (renderElement as RenderElementNode).node;
  }

  private normalizeId(id: string): string {
    return id.replace(/[^a-zA-Z0-9]/g, '_');
  }
}
