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

import {Component, effect, inject, input, OnDestroy, signal} from '@angular/core';
import {ArtifactsApi} from '../../server/artifacts_api';
import type {TapAnchors} from '../../server/contracts/artifacts';

/**
 * The anchors a Session of one LiteRT-LM model captures, as the server describes them (GET
 * /api/artifacts/{id}/anchors), folded away under "Details": what is captured, the anchors
 * skipped with their reasons, and the copy of the model prepared to output them. Only why none
 * is captured, if none is, shows unfolded.
 */
@Component({
  selector: 'anchor-summary',
  templateUrl: './anchor_summary.ng.html',
  styleUrl: './anchor_summary.scss',
})
export class AnchorSummary implements OnDestroy {
  readonly artifact = input.required<string>();
  /** That both sides share one capture of this model, as the server says; '' if they do not. */
  readonly shared = input('');
  readonly anchors = signal<TapAnchors | null>(null);
  readonly error = signal('');
  private readonly api = inject(ArtifactsApi);
  private destroyed = false;
  private request = 0;

  constructor() {
    effect(() => void this.load(this.artifact()));
  }

  ngOnDestroy(): void {
    this.destroyed = true;
  }

  async load(artifact = this.artifact()): Promise<void> {
    const request = ++this.request;
    this.error.set('');
    if (!artifact) {
      this.anchors.set(null);
      return;
    }
    try {
      const anchors = await this.api.tapAnchors(artifact);
      if (request === this.request && !this.destroyed) {
        this.anchors.set(anchors);
      }
    } catch (error) {
      if (request === this.request && !this.destroyed) {
        this.anchors.set(null);
        this.error.set(error instanceof Error ? error.message : String(error));
      }
    }
  }
}
