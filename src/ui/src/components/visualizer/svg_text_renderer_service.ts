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

import {Injectable} from '@angular/core';
import * as d3 from 'd3';

import {FontWeight} from './common/types';
import {LabelData} from './webgl_texts';

/**
 * A service for rendering text using SVG elements.
 */
@Injectable()
export class SvgTextRendererService {
  renderTexts(
    svgElement: SVGElement,
    labels: LabelData[],
    isDarkMode: boolean,
    groupSelector = 'g.root-group',
  ) {
    const group = d3.select(svgElement).select(groupSelector);
    const texts = group.selectAll('text').data(labels, (d: any) => d.id);

    // Enter + Update.
    const textsUpdate = texts
      .enter()
      .append('text')
      .merge(texts as any)
      .attr('x', (d) => d.x)
      .attr('y', (d) => d.z)
      .attr('text-anchor', (d) => {
        switch (d.hAlign) {
          case 'left':
            return 'start';
          case 'right':
            return 'end';
          case 'center':
            return 'middle';
          default:
            return 'middle';
        }
      })
      .attr('dominant-baseline', (d) => {
        switch (d.vAlign) {
          case 'top':
            return 'hanging';
          case 'bottom':
            return 'alphabetic';
          case 'center':
            return 'central';
          default:
            return 'central';
        }
      })
      .style('font-size', (d) => `${d.height}px`)
      .style('font-family', (d) =>
        d.weight === FontWeight.MONOSPACE_MEDIUM
          ? 'Roboto Mono, monospace'
          : 'Google Sans Text, Arial, Helvetica, sans-serif',
      )
      .style('font-weight', (d) =>
        d.weight === FontWeight.BOLD || d.weight === FontWeight.MEDIUM
          ? '700'
          : '400',
      )
      .style('fill', (d) => {
        if (d.color) {
          return `rgb(${Math.round(d.color.r * 255)}, ${Math.round(
            d.color.g * 255,
          )}, ${Math.round(d.color.b * 255)})`;
        }
        return isDarkMode ? 'white' : 'black';
      })
      .attr('transform', (d) => {
        if (d.angle) {
          return `rotate(${(d.angle * 180) / Math.PI}, ${d.x}, ${d.z})`;
        }
        return null;
      })
      .style('pointer-events', 'none');

    textsUpdate.each(function (d) {
      const textElement = d3.select(this);
      const lines = d.label.split('\n');
      const tspans = textElement.selectAll('tspan').data(lines);

      tspans
        .enter()
        .append('tspan')
        .merge(tspans as any)
        .attr('x', d.x)
        .attr('dy', (line, i) => (i === 0 ? 0 : '1.2em'))
        .text((line) => line);

      tspans.exit().remove();
    });

    // Exit.
    texts.exit().remove();
  }

  clear(svgElement: SVGElement, groupSelector = 'g.root-group') {
    d3.select(svgElement).select(groupSelector).selectAll('text').remove();
  }

  clearAll(svgElement: SVGElement) {
    this.clear(svgElement, 'g.node-labels-group');
    this.clear(svgElement, 'g.edge-labels-group');
    this.clear(svgElement, 'g.attrs-table-labels-group');
  }

  setTextsVisible(svgElement: SVGElement, visible: boolean) {
    d3.select(svgElement).style('display', visible ? 'block' : 'none');
  }

  setTextsOpacity(svgElement: SVGElement, nodeIds: string[], opacity: number) {
    const nodeIdSet = new Set(nodeIds);
    d3.select(svgElement)
      .selectAll('text')
      .filter((d: any) => nodeIdSet.has(d.nodeId))
      .style('opacity', opacity);
  }

  restoreTextsOpacity(svgElement: SVGElement) {
    d3.select(svgElement).selectAll('text').style('opacity', 1);
  }

  updateColorInNode(svgElement: SVGElement, nodeIds: string[], color: string) {
    const nodeIdSet = new Set(nodeIds);
    d3.select(svgElement)
      .selectAll('text')
      .filter((d: any) => nodeIdSet.has(d.nodeId))
      .style('fill', color);
  }

  restoreColors(svgElement: SVGElement, isDarkMode: boolean) {
    d3.select(svgElement)
      .selectAll('text')
      .style('fill', (d: any) => {
        if (d.color) {
          return `rgb(${Math.round(d.color.r * 255)}, ${Math.round(
            d.color.g * 255,
          )}, ${Math.round(d.color.b * 255)})`;
        }
        return isDarkMode ? 'white' : 'black';
      });
  }
}
