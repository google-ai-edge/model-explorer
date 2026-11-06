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

import {signal} from '@angular/core';
import {type ComponentFixture, TestBed} from '@angular/core/testing';
import {By} from '@angular/platform-browser';

import {ANALYSES} from '../../../../analyses/registry';
import {CaptureStore} from '../../../../server/capture_store';
import {KvApi} from '../../../../server/kv_api';
import {StageStateInfo} from './stage_state_info';

describe('Details frame', () => {
  let originalRangeBoundingRect: PropertyDescriptor | undefined;

  beforeEach(() => {
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe(): void {}
        disconnect(): void {}
      },
    );
    originalRangeBoundingRect = Object.getOwnPropertyDescriptor(
      Range.prototype,
      'getBoundingClientRect',
    );
    Object.defineProperty(Range.prototype, 'getBoundingClientRect', {
      configurable: true,
      value: () => ({width: 0}),
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    if (originalRangeBoundingRect) {
      Object.defineProperty(Range.prototype, 'getBoundingClientRect', originalRangeBoundingRect);
    } else {
      Reflect.deleteProperty(Range.prototype, 'getBoundingClientRect');
    }
  });

  function create(kind: string): ComponentFixture<StageStateInfo> {
    TestBed.configureTestingModule({
      providers: [
        {provide: KvApi, useValue: {}},
        {provide: CaptureStore, useValue: {captureId: signal('capture'), session: signal(null)}},
      ],
    });
    const fixture = TestBed.createComponent(StageStateInfo);
    fixture.componentRef.setInput('panel', {id: 'analysis-1', kind, turn: 2, stage: 'decode'});
    fixture.componentRef.setInput('label', 'Analysis 0');
    fixture.componentRef.setInput('range', {start: 32, end: 34});
    fixture.detectChanges();
    return fixture;
  }

  it('shows the registry body of the panel analysis and replaces it when the kind changes', () => {
    const fixture = create('tokens');
    const body = () =>
      Object.values(ANALYSES).filter(
        ({details}) => !!fixture.debugElement.query(By.directive(details)),
      );
    expect(body()).toEqual([ANALYSES.final_logits]);
    for (const [kind, analysis] of [
      ['distribution', ANALYSES.sampling],
      ['kv', ANALYSES.kv],
      ['activations', ANALYSES.intermediate_tensors],
    ] as const) {
      fixture.componentRef.setInput('panel', {id: 'analysis-1', kind, turn: 2, stage: 'decode'});
      fixture.detectChanges();
      expect(body()).toEqual([analysis]);
    }
    fixture.componentRef.setInput('panel', {
      id: 'analysis-1',
      kind: 'unknown',
      turn: 2,
      stage: 'decode',
    });
    fixture.detectChanges();
    expect(fixture.componentInstance.body()).toBeUndefined();
    expect(body()).toEqual([]);
    expect(fixture.nativeElement.children).toHaveLength(0);
  });

  it('passes its inputs to the body and the body outputs back out', () => {
    const fixture = create('distribution');
    const body = fixture.debugElement.query(By.directive(ANALYSES.sampling.details))
      .componentInstance as InstanceType<typeof ANALYSES.sampling.details>;
    expect(body.label()).toBe('Analysis 0');
    fixture.componentRef.setInput('range', {start: 33, end: 34});
    fixture.componentRef.setInput('alignment', 'steps');
    fixture.detectChanges();
    expect(body.range()).toEqual({start: 33, end: 34});
    expect(body.alignment()).toBe('steps');
    const located: string[] = [];
    fixture.componentInstance.locateFinal.subscribe((label) => located.push(label));
    fixture.componentInstance.locateAnalysis.subscribe(() => located.push('analysis'));
    body.locateStatistic('Sampling distribution');
    body.locateDefinition({label: 'Name', value: 'Analysis 0'});
    expect(located).toEqual(['Sampling distribution', 'analysis']);
  });
});
