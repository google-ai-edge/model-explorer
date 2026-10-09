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

import {A11yModule} from '@angular/cdk/a11y';
import {CdkConnectedOverlay, OverlayModule} from '@angular/cdk/overlay';
import {
  afterNextRender,
  Component,
  computed,
  inject,
  Injector,
  input,
  output,
  signal,
  viewChild,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import {
  OverlayDialog,
  OverlayPanel,
} from '../../../shared/overlay_dialog/overlay_dialog';
export interface HfFile {
  source: 'huggingface';
  repository: string;
  artifact: string;
  revision: string;
  sourceUrl: string;
}
/** The prototype's openHF flow, hosted by the same CDK overlay used by ConfigPicker. */
@Component({
  selector: 'hf-file-picker',
  imports: [
    OverlayDialog,
    OverlayPanel,
    OverlayModule,
    A11yModule,
    MatButtonModule,
    MatIconModule,
    MatTooltipModule,
  ],
  template: ` <button
      type="button"
      mat-button
      cdkOverlayOrigin
      #origin="cdkOverlayOrigin"
      aria-label="Hugging Face"
      matTooltip="Select a model file from Hugging Face"
      aria-haspopup="dialog"
      [attr.aria-expanded]="open()"
      [disabled]="disabled()"
      (click)="repository.set(''); path.set(''); open.set(true)"
    >
      <mat-icon>cloud_download</mat-icon>Hugging Face
    </button>
    <ng-template
      overlayDialog
      [origin]="origin"
      [open]="open()"
      [gap]="4"
      (closed)="open.set(false)"
    >
      <section class="hf-picker" aria-label="Hugging Face model file" overlayPanel>
        <label
          >Repository or file URL<input
            cdkFocusInitial
            aria-label="Repository or file URL"
            placeholder="org/repo or Hugging Face file URL"
            [value]="repository()"
            (input)="updateRepository($any($event.target).value)"
        /></label>
        @if (repository().trim() && !fileUrl()) {
          <label
            >LiteRT-LM file<input
              aria-label="LiteRT-LM file"
              placeholder="path/to/model.litertlm"
              [value]="path()"
              (input)="path.set($any($event.target).value)"
          /></label>
        }
        <div class="actions">
          <button
            type="button"
            mat-button
            matTooltip="Cancel file selection"
            (click)="open.set(false)"
          >
            Cancel</button
          ><button
            type="button"
            mat-button
            [disabled]="!selection()"
            matTooltip="Use this Hugging Face file as the model source"
            (click)="choose()"
          >
            Use file
          </button>
        </div>
      </section></ng-template
    >`,
  styleUrl: './hf_file_picker.scss',
})
export class HfFilePicker {
  readonly disabled = input(false);
  readonly selected = output<HfFile>();
  readonly open = signal(false);
  readonly repository = signal('');
  readonly path = signal('');
  private readonly overlay = viewChild(CdkConnectedOverlay);
  private readonly injector = inject(Injector);
  updateRepository(value: string) {
    this.repository.set(value);
    // The conditional path field changes the connected panel's size.
    afterNextRender(() => this.overlay()?.overlayRef?.updatePosition(), {
      injector: this.injector,
    });
  }
  readonly fileUrl = computed(() => {
    try {
      const url = new URL(this.repository().trim());
      if (url.protocol !== 'https:' || url.hostname !== 'huggingface.co')
        return null;
      const match = url.pathname.match(
        /^\/([^/]+\/[^/]+)\/resolve\/([^/]+)\/(.+)$/,
      );
      return match
        ? {
            repository: match[1],
            revision: decodeURIComponent(match[2]),
            artifact: decodeURIComponent(match[3]),
            sourceUrl: url.href,
          }
        : null;
    } catch {
      return null;
    }
  });
  readonly selection = computed<HfFile | null>(() => {
    const url = this.fileUrl();
    if (url) return {source: 'huggingface', ...url};
    const repository = this.repository().trim(),
      artifact = this.path().trim();
    if (
      !/^[\w.-]+\/[\w.-]+$/.test(repository) ||
      !artifact ||
      artifact.startsWith('/') ||
      artifact.split('/').includes('..')
    )
      return null;
    return {
      source: 'huggingface',
      repository,
      artifact,
      revision: 'main',
      sourceUrl: '',
    };
  });
  choose() {
    const selected = this.selection();
    if (!selected) return;
    this.open.set(false);
    this.selected.emit(selected);
  }
}
