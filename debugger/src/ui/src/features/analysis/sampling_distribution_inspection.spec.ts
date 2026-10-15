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
import {
  SamplingDistributionInspection,
  samplingTopCandidates,
  samplingCandidateUnion,
} from './sampling_distribution_inspection';
import type {SamplingDistribution} from '../../data/contracts/token_analysis';

function distribution(): SamplingDistribution {
  const side = {candidate_count: 3, basis: null, sampler: null};
  return {
    context: 'different',
    vocabulary_compatible: true,
    rows_complete: true,
    reason: null,
    ref: {...side},
    target: {...side},
    rows: [
      {
        id: 10,
        label: 'a',
        ref: {probability: 0.1, rank: 3, selected: true},
        target: {probability: 0.6, rank: 1, selected: true},
      },
      {
        id: 20,
        label: '<tag>',
        ref: {probability: 0.5, rank: 1, selected: false},
        target: {probability: 0.15, rank: 3, selected: false},
      },
      {
        id: 30,
        label: '\n',
        ref: {probability: 0.4, rank: 2, selected: false},
        target: {probability: 0.25, rank: 2, selected: false},
      },
      {id: 40, label: null, ref: {probability: 0, rank: null, selected: false}, target: null},
    ],
  };
}

describe('Sampling distribution Top K inspection', () => {
  beforeEach(() => {
    Object.defineProperty(Range.prototype, 'getBoundingClientRect', {
      configurable: true,
      value: () => ({width: 0}),
    });
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe() {}
        disconnect() {}
      },
    );
    vi.stubGlobal('matchMedia', () => ({
      matches: true,
      addEventListener() {},
      removeEventListener() {},
    }));
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    Reflect.deleteProperty(Range.prototype, 'getBoundingClientRect');
  });
  function create() {
    const fixture = TestBed.createComponent(SamplingDistributionInspection);
    for (const [key, value] of Object.entries({
      singlePosition: true,
      refDistribution: distribution(),
      targetDistribution: distribution(),
    }))
      fixture.componentRef.setInput(key, value);
    return fixture;
  }
  it('selects each side independently by probability without renormalizing or appending an out-of-top-K selected token', () => {
    const source = distribution();
    expect(samplingTopCandidates(source, 'ref', 2).map((row) => [row.id, row.probability])).toEqual(
      [
        [20, 0.5],
        [30, 0.4],
      ],
    );
    expect(samplingTopCandidates(source, 'target', 2).map((row) => [row.id, row.selected])).toEqual(
      [
        [10, true],
        [30, false],
      ],
    );
    expect(samplingTopCandidates(source, 'ref', 10)).toHaveLength(3);
    expect(samplingTopCandidates(undefined, 'ref', 10)).toEqual([]);
    expect(source.rows.map((row) => row.id)).toEqual([10, 20, 30, 40]);
  });
  it('shows four columns for a single position and clears rows for multiple positions', () => {
    const fixture = create();
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    expect(
      Array.from(el.querySelectorAll('th')).map((cell) =>
        cell.querySelector('explorer-info-value .value')?.textContent?.trim(),
      ),
    ).toEqual(['Token ID', 'Ref Prob', 'Target Prob', 'Token']);
    expect(el.querySelector('th:last-child')?.getAttribute('colspan')).toBe('2');
    expect(
      Array.from(el.querySelectorAll('tbody tr td:last-child')).every(
        (cell) => cell.getAttribute('colspan') === '2',
      ),
    ).toBe(true);
    expect(el.querySelectorAll('tbody tr')).toHaveLength(3);
    expect(el.querySelector('tag')).toBeNull();
    expect(el.textContent).toContain('"<tag>"');
    fixture.componentRef.setInput('singlePosition', false);
    fixture.detectChanges();
    expect(el.querySelectorAll('tbody tr')).toHaveLength(1);
    expect(
      Array.from(el.querySelectorAll('tbody td')).map((cell) => cell.textContent?.trim()),
    ).toEqual(['-', '-', '-', '-']);
    fixture.destroy();
  });
  it('takes the union of independently ranked Top 64 regardless of sampler Top-k', () => {
    const fixture = create();
    const source = distribution();
    source.rows = Array.from({length: 160}, (_, id) => ({
      id,
      label: String(id),
      ref: {probability: 1 / (id + 2), rank: id + 1, selected: false},
      target: {probability: 1 / (161 - id), rank: 160 - id, selected: false},
    }));
    expect(samplingCandidateUnion(source, source)).toHaveLength(128);
    source.rows = source.rows.map((row) => ({...row, target: {...row.ref!}}));
    source.ref.sampler = {top_k: 1, temperature: null, top_p: null, seed: null};
    fixture.componentRef.setInput('refDistribution', source);
    fixture.componentRef.setInput('targetDistribution', source);
    expect(fixture.componentInstance.candidates()).toHaveLength(64);
    expect(fixture.componentInstance.candidates().at(-1)?.id).toBe(63);
    const shifted = {...source, rows: source.rows.slice(32)};
    expect(samplingCandidateUnion(source, shifted)).toHaveLength(96);
    expect(samplingCandidateUnion(undefined, source)).toHaveLength(64);
    expect(samplingCandidateUnion({...source, vocabulary_compatible: false}, source)).toHaveLength(
      128,
    );
    fixture.destroy();
  });
  it('keeps other-side probabilities outside Top 64 and distinguishes missing entries from proven zeros', () => {
    const source = distribution();
    const partial = {
      ...distribution(),
      rows: distribution().rows.filter((row) => row.id !== 10),
      rows_complete: false,
    };
    expect(samplingCandidateUnion(partial, source).find((row) => row.id === 10)?.ref).toBeNull();
    partial.rows_complete = true;
    partial.ref = {...partial.ref, entropy: 1};
    expect(
      samplingCandidateUnion(partial, source).find((row) => row.id === 10)?.ref?.probability,
    ).toBe(0);
    const many = {
      ...source,
      rows: Array.from({length: 128}, (_, id) => ({
        id,
        label: String(id),
        ref: {probability: (128 - id) / 10000, rank: id + 1, selected: false},
        target: {probability: (id + 1) / 10000, rank: 128 - id, selected: false},
      })),
    };
    expect(samplingCandidateUnion(many, many).find((row) => row.id === 127)?.ref?.probability).toBe(
      0.0001,
    );
  });
  it('sorts both probability columns in both directions without changing membership', () => {
    const fixture = create();
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    const ids = () =>
      Array.from(el.querySelectorAll('tbody tr')).map((row) => row.getAttribute('data-token-id'));
    expect(ids()).toEqual(['20', '30', '10']);
    el.querySelector<HTMLButtonElement>('[aria-label="Sort by Target Prob"]')!.click();
    fixture.detectChanges();
    expect(ids()).toEqual(['10', '30', '20']);
    expect(el.querySelectorAll('th')[2].getAttribute('aria-sort')).toBe('descending');
    el.querySelector<HTMLButtonElement>('[aria-label="Sort by Target Prob"]')!.click();
    fixture.detectChanges();
    expect(ids()).toEqual(['20', '30', '10']);
    el.querySelector<HTMLButtonElement>('[aria-label="Sort by Ref Prob"]')!.click();
    fixture.detectChanges();
    expect(ids()).toEqual(['20', '30', '10']);
    el.querySelector<HTMLButtonElement>('[aria-label="Sort by Ref Prob"]')!.click();
    fixture.detectChanges();
    expect(ids()).toEqual(['10', '30', '20']);
    const source = distribution();
    source.rows.forEach((row) => {
      if (row.ref) row.ref.probability = row.ref.probability > 0 ? 0.2 : 0;
    });
    fixture.componentRef.setInput('refDistribution', source);
    fixture.detectChanges();
    expect(ids()).toEqual(['10', '20', '30']);
    expect(source.rows.map((row) => row.id)).toEqual([10, 20, 30, 40]);
    fixture.destroy();
  });
});
