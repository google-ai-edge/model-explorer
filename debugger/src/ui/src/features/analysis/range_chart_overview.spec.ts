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

import {stackRangeOverview} from './range_chart_overview';

describe('stacked Plotly range overview', () => {
  it('separates matched-axis data nested in the same rangeplot and preserves the native brush', () => {
    const host = document.createElement('div');
    host.innerHTML = `<svg><g class="rangeslider-container">
      <rect class="rangeslider-bg" height="72" />
      <g class="rangeslider-rangeplot xy"><g class="overplot">
        <g class="xy"><path class="js-line" /></g>
        <g class="xy2"><path class="js-line" /></g>
      </g></g>
      <g class="rangeslider-rangeplot xy2"></g>
      <rect class="rangeslider-slidebox" height="72" />
    </g></svg>`;
    const brush = host.querySelector('.rangeslider-slidebox');
    stackRangeOverview(host, ['Key', 'Value']);
    const lanes = host.querySelectorAll('.overplot > g');
    expect(lanes[0].getAttribute('transform')).toBe('translate(0,0) scale(1,0.4444444444444444)');
    expect(lanes[1].getAttribute('transform')).toBe('translate(0,40) scale(1,0.4444444444444444)');
    expect(host.querySelector('.rangeslider-slidebox')).toBe(brush);
    expect(brush?.getAttribute('height')).toBe('72');
    stackRangeOverview(host, ['Key', 'Value']);
    expect(host.querySelectorAll('.range-overview-labels')).toHaveLength(1);
    expect(lanes[1].getAttribute('transform')).toBe('translate(0,40) scale(1,0.4444444444444444)');
    stackRangeOverview(host, []);
    expect(lanes[0].hasAttribute('transform')).toBe(false);
    expect(host.querySelector('.range-overview-labels')).toBeNull();
  });
});
