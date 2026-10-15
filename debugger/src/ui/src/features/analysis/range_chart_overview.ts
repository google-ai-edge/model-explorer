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

/**
 * Plotly 4 draws every range-slider subplot over the full preview height,
 * ignoring the main y-axis domains. Keep its native brush and range events,
 * but place the preview plots in separate lanes after each Plotly redraw.
 * This adapter is shared by metric and tensor charts; no trace data is changed.
 */
export function stackRangeOverview(host: HTMLElement, labels: readonly string[]) {
  const slider = host.querySelector<SVGGElement>('.rangeslider-container');
  if (!slider) return;
  const plots = Array.from(slider.querySelectorAll<SVGGElement>(':scope > .rangeslider-rangeplot'));
  slider.querySelector('.range-overview-labels')?.remove();
  const height = Number(slider.querySelector('.rangeslider-bg')?.getAttribute('height'));
  if (plots.length < 2 || labels.length !== plots.length || !(height > 0)) {
    plots.forEach((plot) => plot.removeAttribute('transform'));
    slider.querySelectorAll('.overplot > g').forEach((data) => data.removeAttribute('transform'));
    return;
  }
  const gap = 8;
  const lane = (height - gap * (plots.length - 1)) / plots.length;
  const ns = 'http://www.w3.org/2000/svg';
  const captions = document.createElementNS(ns, 'g');
  captions.setAttribute('class', 'range-overview-labels');
  captions.setAttribute('pointer-events', 'none');
  plots.forEach((plot) => plot.removeAttribute('transform'));
  plots.forEach((plot, index) => {
    const top = index * (lane + gap);
    // Matched axes can put both data groups in the first rangeplot, leaving
    // the second rangeplot empty. Transform the actual subplot data groups.
    const subplot = Array.from(plot.classList).find((name) => name !== 'rangeslider-rangeplot');
    if (subplot) {
      slider
        .querySelectorAll<SVGGElement>(`.overplot > g.${subplot}`)
        .forEach((data) =>
          data.setAttribute('transform', `translate(0,${top}) scale(1,${lane / height})`),
        );
    }
    const label = document.createElementNS(ns, 'text');
    label.setAttribute('x', '-8');
    label.setAttribute('y', String(top + lane / 2));
    label.setAttribute('text-anchor', 'end');
    label.setAttribute('dominant-baseline', 'middle');
    label.setAttribute('fill', 'currentColor');
    label.setAttribute('font-size', '10');
    label.textContent = labels[index];
    captions.append(label);
  });
  slider.append(captions);
}
