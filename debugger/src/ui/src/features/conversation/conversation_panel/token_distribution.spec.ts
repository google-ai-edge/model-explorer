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

import {TestBed} from '@angular/core/testing';
import type {
  SamplingDistribution,
  SamplingDistributionRow,
  SamplingDistributionSide,
  TokenAnalysisPair,
} from '../../../data/contracts/token_analysis';
import {TokenDistribution} from './token_distribution';

const side = (patch: Partial<SamplingDistributionSide> = {}): SamplingDistributionSide => ({
  entropy: 0,
  selected_id: 7,
  selected_rank: 1,
  selected_probability: 1,
  margin: 1,
  candidate_count: 1,
  size: 262144,
  source_forward_id: 0,
  basis: 'reconstructed_greedy',
  sampler: {temperature: 0, top_k: 1, top_p: 1, seed: 0},
  reason: null,
  ...patch,
});
const row = (id: number, ref: number | null, target = ref): SamplingDistributionRow => ({
  id,
  label: ' token ' + id,
  ref: ref === null ? null : {probability: ref, rank: ref === 0 ? null : 1, selected: id === 7},
  target:
    target === null
      ? null
      : {probability: target, rank: target === 0 ? null : 1, selected: id === 8},
});
function analysis(patch: Partial<SamplingDistribution> = {}): TokenAnalysisPair {
  return {
    ref_step: 0,
    target_step: 0,
    metrics: Object.fromEntries(
      ['kl', 'js', 'relative_l2', 'cosine_distance', 'norm_ratio', 'max_abs_error'].map((key) => [
        key,
        {value: null, reason: 'Unavailable'},
      ]),
    ) as TokenAnalysisPair['metrics'],
    distribution: {
      compatible: true,
      reason: null,
      ref: {entropy: 9, selected_probability: 0.08, margin: 0.01},
      target: {entropy: 8, selected_probability: 0.12, margin: 0.02},
      rows: [],
    },
    sampling_distribution: {
      context: 'same',
      vocabulary_compatible: true,
      rows_complete: true,
      reason: null,
      ref: side(),
      target: side({selected_id: 8}),
      rows: [row(7, 1, 0), row(8, 0, 1)],
      ...patch,
    },
  };
}

describe('Sampling Distribution position details', () => {
  const rangeRect = Object.getOwnPropertyDescriptor(Range.prototype, 'getBoundingClientRect');
  let textWidth = 0;
  beforeEach(() => {
    textWidth = 0;
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe() {}
        disconnect() {}
      },
    );
    vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue({
      x: 0,
      y: 0,
      top: 0,
      left: 0,
      right: 60,
      bottom: 20,
      width: 60,
      height: 20,
      toJSON() {},
    });
    Object.defineProperty(Range.prototype, 'getBoundingClientRect', {
      configurable: true,
      value: () => ({width: textWidth}),
    });
  });
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    if (rangeRect) Object.defineProperty(Range.prototype, 'getBoundingClientRect', rangeRect);
    else Reflect.deleteProperty(Range.prototype, 'getBoundingClientRect');
  });
  function create(record = analysis()) {
    const fixture = TestBed.createComponent(TokenDistribution);
    fixture.componentRef.setInput('analysis', record);
    fixture.detectChanges();
    fixture.detectChanges();
    return {fixture, view: fixture.componentInstance, root: fixture.nativeElement as HTMLElement};
  }
  it('does not substitute existing raw-logit softmax data when sampling evidence is absent', () => {
    const record = analysis();
    delete record.sampling_distribution;
    const {view, root} = create(record);
    expect(view.hasDistribution()).toBe(false);
    expect(view.sortedRows()).toEqual([]);
    expect(view.summaryValue('ref', 'selected_id')).toBe('—');
    expect(root.textContent).toContain('Sampling distribution unavailable');
    expect(root.querySelector('table[aria-label="Sampling candidate probabilities"]')).toBeNull();
    expect(root.textContent).not.toContain('Source forward');
  });
  it('keeps zero probability distinct from missing sides and identifies the generated token', () => {
    const {view, root} = create(analysis({rows: [row(7, 1, 0), row(8, null, 1)]}));
    expect(view.summaryValue('ref', 'candidate_count')).toBe('1');
    expect(view.summaryValue('ref', 'selected_rank')).toBe('#1');
    expect(view.probability(view.rows()[0].target)).toBe('0.00%');
    expect(view.probability(view.rows()[1].ref)).toBe('—');
    expect(view.probabilitySegments(view.rows()[0], 'target')[0].help?.rows).toContainEqual({
      label: 'Rank',
      value: 'Outside sampling support',
    });
    expect(root.querySelector('[data-candidate-id="7"] [data-selected-side="ref"]')).not.toBeNull();
    expect(
      root.querySelector('[data-candidate-id="8"] [data-selected-side="target"]'),
    ).not.toBeNull();
    expect(view.tokenSegments(view.rows()[0])[0].help).toEqual({
      title: 'Token',
      text: ' token 7',
      rows: [],
    });
  });
  it.each(['ref', 'target'] as const)(
    'does not borrow a candidate label for an unavailable %s side without vocabulary compatibility',
    (unavailableSide) => {
      const availableSide = unavailableSide === 'ref' ? 'target' : 'ref';
      const record = analysis({
        vocabulary_compatible: false,
        ref: side(),
        target: side(),
        [unavailableSide]: side({
          entropy: undefined,
          selected_probability: null,
          selected_rank: null,
          margin: null,
          candidate_count: null,
          basis: null,
          reason: 'Sampling evidence not captured',
        }),
        rows: [
          row(7, unavailableSide === 'ref' ? null : 1, unavailableSide === 'target' ? null : 1),
        ],
      });
      const {fixture, view} = create(record);
      expect(view.summarySegments(availableSide, 'selected_id')[0].help?.text).toBe(' token 7');
      expect(view.summarySegments(unavailableSide, 'selected_id')[0].help?.text).toBe(
        'Text not captured',
      );
      fixture.componentRef.setInput(
        'analysis',
        analysis({...record.sampling_distribution!, vocabulary_compatible: true}),
      );
      fixture.detectChanges();
      expect(view.summarySegments(unavailableSide, 'selected_id')[0].help?.text).toBe(' token 7');
    },
  );
  it('sorts all candidates before slicing the page and resets to the first page after sorting', () => {
    const source = Array.from({length: 34}, (_, index) => row(index, (34 - index) / 100));
    const {view} = create(analysis({rows: source}));
    expect(view.pages()).toBe(3);
    view.changePage(1);
    view.sortBy('ref');
    expect(view.currentPage()).toBe(0);
    expect(view.rows().map((r) => r.id)).toEqual(Array.from({length: 16}, (_, i) => 33 - i));
    view.changePage(1);
    expect(view.rows().map((r) => r.id)).toEqual(Array.from({length: 16}, (_, i) => 17 - i));
    view.changePage(1);
    expect(view.rows().map((r) => r.id)).toEqual([1, 0]);
    view.sortBy('ref');
    expect(view.rows()[0].id).toBe(0);
    expect(source[0].id).toBe(0);
  });
  it('sorts both probability columns numerically with missing values last and stable ID ties', () => {
    const {view} = create(
      analysis({rows: [row(9, null, 1), row(2, 0.5, 0.5), row(1, 0.5, 0), row(4, 0, null)]}),
    );
    view.sortBy('ref');
    expect(view.rows().map((r) => r.id)).toEqual([4, 1, 2, 9]);
    view.sortBy('ref');
    expect(view.rows().map((r) => r.id)).toEqual([1, 2, 4, 9]);
    view.sortBy('target');
    expect(view.rows().map((r) => r.id)).toEqual([1, 2, 9, 4]);
    view.sortBy('id');
    expect(view.rows().map((r) => r.id)).toEqual([1, 2, 4, 9]);
    view.sortBy('id');
    expect(view.rows().map((r) => r.id)).toEqual([9, 4, 2, 1]);
  });
  it('resets pagination when the inspected position changes and retains single-page controls', () => {
    const {fixture, view, root} = create(
      analysis({rows: Array.from({length: 34}, (_, id) => row(id, 0.1))}),
    );
    view.changePage(2);
    expect(view.currentPage()).toBe(2);
    fixture.componentRef.setInput('analysis', {...analysis(), ref_step: 1, target_step: 1});
    fixture.detectChanges();
    expect(view.currentPage()).toBe(0);
    const pager = root.querySelector('explorer-pagination')!;
    expect(pager.textContent).toContain('1 / 1');
    expect(Array.from(pager.querySelectorAll('button')).every((button) => button.disabled)).toBe(
      true,
    );
  });
  it('keeps candidate details in one numbered group without duplicating shared statistics or metadata', () => {
    const {root} = create();
    expect(
      Array.from(root.children).map((element) =>
        element.querySelector('section')?.getAttribute('aria-label'),
      ),
    ).toEqual(['Candidates']);
    expect(root.querySelector('section > header strong')?.textContent).toBe('Candidates (1)');
    expect(root.querySelector('.item-heading .item-index')?.textContent).toBe('0');
    expect(root.querySelector('.item-heading .item-name')?.textContent).toBe(
      'Sampling distribution',
    );
    expect(root.querySelector('.item-heading explorer-pagination')).not.toBeNull();
    expect(
      Array.from(root.querySelectorAll('.summary-table tbody th')).map((cell) => cell.textContent),
    ).toEqual(['Selected token', 'Candidate count', 'Selected rank']);
    expect(root.textContent).not.toContain('Entropy');
    expect(root.textContent).not.toContain('Sampling Metadata');
    const cells = Array.from(root.querySelectorAll('.candidate-value'));
    expect(cells).toHaveLength(4);
    expect(cells.every((cell) => cell.querySelector('.selected-marker-slot'))).toBe(true);
    expect(root.querySelectorAll('.selected-marker-slot mat-icon')).toHaveLength(2);
  });
  it('uses the same group for a range without showing stale point details or active page controls', () => {
    const {fixture, view, root} = create(
      analysis({rows: Array.from({length: 34}, (_, id) => row(id, 0.1))}),
    );
    view.changePage(2);
    fixture.componentRef.setInput('singlePosition', false);
    fixture.detectChanges();
    expect(root.querySelector('.item-heading .item-name')?.textContent).toBe(
      'Sampling distribution',
    );
    expect(root.textContent).toContain('Select one position in the line chart');
    expect(root.querySelector('.summary-table')).toBeNull();
    expect(root.querySelector('.candidate-table')).toBeNull();
    const pager = root.querySelector('.item-heading explorer-pagination')!;
    expect(pager.textContent).toContain('1 / 1');
    expect(Array.from(pager.querySelectorAll('button')).every((button) => button.disabled)).toBe(
      true,
    );
  });
  it('hides previous candidate details while new sampling data loads or fails and restores them on recovery', () => {
    const {fixture, root} = create();
    fixture.componentRef.setInput('loading', true);
    fixture.detectChanges();
    expect(root.querySelector('.summary-table')).toBeNull();
    expect(root.querySelector('.candidate-table')).toBeNull();
    expect(root.textContent).toContain('Loading sampling distribution…');
    fixture.componentRef.setInput('loading', false);
    fixture.componentRef.setInput('error', 'Request failed');
    fixture.detectChanges();
    expect(root.querySelector('.summary-table')).toBeNull();
    expect(root.querySelector('.candidate-table')).toBeNull();
    expect(root.textContent).toContain('Sampling distribution unavailable: Request failed');
    fixture.componentRef.setInput('error', '');
    fixture.detectChanges();
    expect(root.querySelector('.summary-table')).not.toBeNull();
    expect(root.querySelector('.candidate-table')).not.toBeNull();
  });
  it('retains the candidate ID width while the next data is pending or fails', () => {
    const {fixture, view} = create(analysis({rows: [row(123456789012, 1)]}));
    const width = view.identityWidth();
    expect(width).toBeGreaterThan(64);
    fixture.componentRef.setInput('loading', true);
    fixture.componentRef.setInput('analysis', undefined);
    fixture.detectChanges();
    expect(view.identityWidth()).toBe(width);
    fixture.componentRef.setInput('loading', false);
    fixture.componentRef.setInput('error', 'Request failed');
    fixture.detectChanges();
    expect(view.identityWidth()).toBe(width);
    fixture.componentRef.setInput('error', '');
    fixture.componentRef.setInput('analysis', analysis());
    fixture.detectChanges();
    expect(view.identityWidth()).toBe(64);
  });
  it('exposes explicit inline expansion for clipped candidate values without fake locator buttons', () => {
    textWidth = 120;
    const {fixture, root} = create();
    const expand = root.querySelector<HTMLButtonElement>(
      'button[aria-label="Expand Reference token 7 probability"]',
    );
    expect(expand).not.toBeNull();
    expand!.click();
    fixture.detectChanges();
    expect(
      root.querySelector('button[aria-label="Collapse Reference token 7 probability"]'),
    ).not.toBeNull();
    expect(root.querySelector('[explorer-locator]')).toBeNull();
  });
});
