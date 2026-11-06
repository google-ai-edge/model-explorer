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
import {TestBed} from '@angular/core/testing';
import {MAT_DIALOG_DATA} from '@angular/material/dialog';
import {CaptureStore} from '../../server/capture_store';
import type {Session} from '../../server/contracts/capture';
import {RuntimeService} from '../../server/runtime_service';
import {SessionListService} from '../../server/session_list_service';
import {session} from '../../server/testing/session_fixtures';
import {SessionConfiguration} from './session_configuration';

describe('SessionConfiguration', () => {
  // The configuration shows each side's runner version: what the attached capture recorded when
  // it is a chat of this Session, else the version the side chose.
  it('the configuration shows each side runner version, recorded or chosen', () => {
    const kept = '61dcfc77225019e067580dceb3ea268483d1938c';
    const other = 'a484d21d5113886348fe4c56af61830e45e664a8';
    const configured = session('s', {
      runs: [
        {id: 'ref', runtime: 'LiteRT-LM', device: 'local', runnerBuild: kept},
        {id: 'target', runtime: 'LiteRT-LM', device: 'local', runnerBuild: 'dev'},
      ],
    });
    const captured = signal<Session | null>(null);
    TestBed.configureTestingModule({
      providers: [
        {provide: SessionListService, useValue: {items: () => [configured]}},
        {provide: CaptureStore, useValue: {session: captured}},
        // The options the server describes; none are needed for the runner versions.
        {provide: RuntimeService, useValue: {capability: () => undefined}},
        {provide: MAT_DIALOG_DATA, useValue: configured},
      ],
    });
    const view = TestBed.runInInjectionContext(() => new SessionConfiguration());
    expect(view.runnerVersion('ref')).toBe('LiteRT-LM 61dcfc772250');
    expect(view.runnerCommit('ref')).toBe(kept);
    expect(view.runnerVersion('target')).toBe('Development build');
    expect(view.runnerCommit('target')).toBe('');
    const runs = [
      {id: 'ref', provenance: {runtime_version: kept}},
      {id: 'target', provenance: {runtime_version: other}},
    ];
    captured.set({capture_id: 's', runs} as unknown as Session);
    expect(view.runnerVersion('target')).toBe('LiteRT-LM a484d21d5113 · Development build');
    expect(view.runnerCommit('target')).toBe(other);
    captured.set({capture_id: 'elsewhere', runs} as unknown as Session);
    expect(view.runnerVersion('target'), "another Session's capture").toBe('Development build');
  });
});
