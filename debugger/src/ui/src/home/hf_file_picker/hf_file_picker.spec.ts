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
import {ArtifactsApi, type HuggingFaceRequest} from '../../server/artifacts_api';
import {HfFilePicker} from './hf_file_picker';

describe('HfFilePicker', () => {
  it('sends a link, or a repository with its file and revision, for the server to check', () => {
    const hf = TestBed.runInInjectionContext(() => new HfFilePicker());
    expect(hf.selection(), 'nothing named').toBe(null);
    hf.link.set('gemma');
    expect(hf.selection(), 'a repository id is <owner>/<name>').toBe(null);
    hf.link.set('litert-community/gemma-4-E2B-it-litert-lm');
    hf.file.set(' gemma-4-E2B-it.litertlm ');
    hf.revision.set('');
    expect(hf.selection()).toEqual({
      link: 'litert-community/gemma-4-E2B-it-litert-lm',
      file: 'gemma-4-E2B-it.litertlm',
      revision: '',
    });
    hf.link.set(
      'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/blob/main/gemma-4-E2B-it.litertlm',
    );
    expect(hf.namesFile() && hf.namesRevision(), 'the link names both').toBe(true);
    expect(hf.selection()?.file, 'what the link names wins').toBe('');
    hf.link.set('https://huggingface.co/google/gemma-4-E2B-it');
    hf.revision.set('3e22461f65e89153144f8adb70e3b8c2cc9845a7');
    expect(hf.namesFile()).toBe(false);
    expect(hf.selection()?.revision).toBe('3e22461f65e89153144f8adb70e3b8c2cc9845a7');
    const sent: unknown[] = [];
    hf.selected.subscribe((value) => sent.push(value));
    hf.open.set(true);
    hf.choose();
    expect(hf.open()).toBe(false);
    expect(sent).toHaveLength(1);
  });

  it('says what the server has of the model named, once the fields rest', async () => {
    vi.useFakeTimers();
    const asked: HuggingFaceRequest[] = [];
    TestBed.configureTestingModule({
      providers: [
        {
          provide: ArtifactsApi,
          useValue: {
            checkHuggingFace: async (request: HuggingFaceRequest) => {
              asked.push(request);
              return {on_server: true, size: 10.3e9, text: 'On the Server · 10.3 GB'};
            },
          },
        },
      ],
    });
    const hf = TestBed.runInInjectionContext(() => new HfFilePicker());
    hf.show();
    hf.link.set('org/repo');
    TestBed.tick();
    expect(hf.status(), 'not before the fields rest').toBe('');
    await vi.advanceTimersByTimeAsync(500);
    expect(asked).toEqual([{runtime: 'LiteRT-LM', link: 'org/repo', file: '', revision: ''}]);
    expect(hf.status()).toBe('On the Server · 10.3 GB');
    hf.link.set('nothing');
    TestBed.tick();
    expect(hf.status(), 'nothing named, nothing said').toBe('');
    vi.useRealTimers();
  });
});
