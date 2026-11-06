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

import {LatestRequest} from './latest_request';

describe('LatestRequest', () => {
  it('keeps the last answer while a newer request loads or fails, until nothing is asked', async () => {
    const request = new LatestRequest<string>();
    request.load(async () => 'first');
    await vi.waitFor(() => expect(request.data()).toBe('first'));
    expect(request.last()).toBe('first');

    let fail!: (error: Error) => void;
    request.load(() => new Promise<string>((_, reject) => (fail = reject)));
    expect(request.data()).toBeNull();
    expect(request.loading()).toBe(true);
    expect(request.last()).toBe('first');
    fail(new Error('refused'));
    await vi.waitFor(() => expect(request.error()).toBe('refused'));
    expect(request.last()).toBe('first');

    request.load(null);
    expect(request.last()).toBeNull();
    expect(request.loading()).toBe(false);
  });

  it('takes the last answer again after a newer request failed, and forgets its error', async () => {
    const request = new LatestRequest<string>();
    request.load(async () => 'first');
    await vi.waitFor(() => expect(request.data()).toBe('first'));
    request.load(async () => {
      throw new Error('refused');
    });
    await vi.waitFor(() => expect(request.error()).toBe('refused'));
    request.reuse();
    expect([request.data(), request.last(), request.error(), request.loading()]).toEqual([
      'first',
      'first',
      '',
      false,
    ]);

    // A newer request still loading is cancelled: its answer is never published.
    let late!: (value: string) => void;
    request.load(() => new Promise<string>((resolve) => (late = resolve)));
    request.reuse();
    late('newer');
    await Promise.resolve();
    expect([request.data(), request.loading()]).toEqual(['first', false]);
  });

  it('never publishes an answer that came after a newer request', async () => {
    const request = new LatestRequest<string>();
    let late!: (value: string) => void;
    request.load(() => new Promise<string>((resolve) => (late = resolve)));
    request.load(async () => 'newer');
    late('older');
    await vi.waitFor(() => expect(request.data()).toBe('newer'));
    await Promise.resolve();
    expect(request.last()).toBe('newer');
  });
});
