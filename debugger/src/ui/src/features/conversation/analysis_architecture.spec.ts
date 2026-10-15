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
import {AnalysisArchitecture} from './analysis_architecture';

describe('Analysis architecture picker', () => {
  function create(stage: 'prefill' | 'decode' = 'decode') {
    const fixture = TestBed.createComponent(AnalysisArchitecture);
    fixture.componentRef.setInput('model', 'Captured model');
    fixture.componentRef.setInput('stage', stage);
    fixture.detectChanges();
    return fixture;
  }

  it('shows the captured model and phase, with no initial object or mode selection', () => {
    const fixture = create();
    expect(fixture.nativeElement.textContent).toContain('Captured model');
    expect(fixture.nativeElement.textContent).toContain('Decode');
    expect(fixture.nativeElement.querySelectorAll('button[aria-pressed="true"]').length).toBe(0);
    expect(fixture.nativeElement.querySelectorAll('input,select').length).toBe(0);
    expect(fixture.nativeElement.querySelectorAll('button').length).toBe(4);
  });

  it('places one Intermediate Tensors entry inside Transformer layers and keeps KV cache outside', () => {
    const fixture = create();
    const emit = vi.fn();
    fixture.componentInstance.selectionChange.subscribe(emit);
    const buttons = fixture.nativeElement.querySelectorAll(
      '[aria-label="Analyze Intermediate Tensors"]',
    );
    expect(buttons.length).toBe(1);
    const button = buttons[0];
    const layers = fixture.nativeElement.querySelector('[data-module="layers"]');
    expect(button.closest('.module')).toBe(layers);
    expect(
      fixture.nativeElement.querySelector('[aria-label="Analyze KV cache"]').closest('.module'),
    ).toBeNull();
    expect(fixture.nativeElement.textContent).not.toContain('Layer Activation');
    button.click();
    expect(emit).toHaveBeenCalledWith('activations');
    fixture.componentRef.setInput('selected', 'activations');
    fixture.detectChanges();
    expect(button.getAttribute('aria-pressed')).toBe('true');
    expect(layers.classList.contains('related')).toBe(true);
  });

  it('omits generation modules and objects for Prefill', () => {
    const fixture = create('prefill');
    expect(fixture.nativeElement.textContent).toContain('Prompt tokens');
    expect(fixture.nativeElement.textContent).not.toContain('LM head');
    expect(fixture.nativeElement.querySelectorAll('button').length).toBe(2);
    expect(fixture.nativeElement.querySelector('[aria-label="Analyze Logits"]')).toBeNull();
  });

  it('emits an object choice and highlights its associated module without changing topology', () => {
    const fixture = create();
    const emit = vi.fn();
    fixture.componentInstance.selectionChange.subscribe(emit);
    const button = fixture.nativeElement.querySelector('[aria-label="Analyze KV cache"]');
    button.click();
    expect(emit).toHaveBeenCalledWith('kv');
    fixture.componentRef.setInput('selected', 'kv');
    fixture.detectChanges();
    expect(button.getAttribute('aria-pressed')).toBe('true');
    expect(fixture.nativeElement.querySelector('.module.related').textContent).toContain(
      'Transformer layers',
    );
    expect(fixture.nativeElement.querySelectorAll('button').length).toBe(4);
  });

  it('reports current availability without inventing a captured value', () => {
    const fixture = create();
    fixture.componentRef.setInput('selected', 'kv');
    fixture.componentRef.setInput('unavailable', {kv: 'Checking captured KV…'});
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('[role="status"]').textContent).toContain(
      'Checking captured KV',
    );
    fixture.componentRef.setInput('unavailable', {kv: 'KV not captured'});
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('[role="status"]').textContent).toBe(
      'KV not captured',
    );
    fixture.componentRef.setInput('unavailable', {});
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('[role="status"]')).toBeNull();
  });

  it('clears on a blank click but not after dragging or clicking a module', () => {
    const fixture = create();
    const emit = vi.fn();
    fixture.componentInstance.selectionChange.subscribe(emit);
    const canvas = fixture.nativeElement.querySelector('.canvas');
    const pointer = (element: Element, type: string, x: number) =>
      element.dispatchEvent(
        new MouseEvent(type, {bubbles: true, clientX: x, clientY: 5, button: 0}),
      );
    pointer(canvas, 'pointerdown', 10);
    pointer(canvas, 'pointerup', 10);
    expect(emit).toHaveBeenCalledWith(null);
    emit.mockClear();
    pointer(canvas, 'pointerdown', 10);
    pointer(canvas, 'pointerup', 30);
    expect(emit).not.toHaveBeenCalled();
    const module = fixture.nativeElement.querySelector('.module');
    pointer(module, 'pointerdown', 10);
    pointer(module, 'pointerup', 10);
    expect(emit).not.toHaveBeenCalled();
  });
});
