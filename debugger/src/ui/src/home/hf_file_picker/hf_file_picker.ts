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
  effect,
  inject,
  Injector,
  input,
  output,
  signal,
  viewChild,
  type WritableSignal,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import {ArtifactsApi} from '../../server/artifacts_api';
import type {HuggingFaceForm} from '../../server/contracts/runtime';
import {OverlayDialog, OverlayPanel} from '../../widgets/overlay_dialog/overlay_dialog';

/** How long the fields rest before the server is asked whether it has the model. */
const CHECK_DELAY_MS = 400;

/** A model named on Hugging Face, as the editor sends it to the server, which checks it. */
export interface HfModel {
  link: string;
  file: string;
  revision: string;
}

/**
 * Names a side's model on Hugging Face: a link or repository id, the file of a LiteRT-LM
 * repository unless the link names one, and a revision. Its words are the server's form of the
 * side's runtime (`form`); the server checks what it names, resolves and fetches it. Before it is
 * chosen, the server says whether it has the model already, and its size ("On the Server ·
 * 10.3 GB", "Downloads to the Server · 15.0 GB"). Hosted by the same CDK overlay as
 * ConfigPicker.
 */
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
      data-source-hf
      [attr.aria-label]="'Hugging Face' + (side() ? ' · ' + side() : '')"
      [matTooltip]="form()?.help ?? ''"
      aria-haspopup="dialog"
      [attr.aria-expanded]="open()"
      [disabled]="disabled() || !form()"
      (click)="show()"
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
      @if (form(); as texts) {
        <section class="hf-picker" aria-label="Hugging Face model" overlayPanel>
          <label
            >{{ texts.link_label
            }}<input
              cdkFocusInitial
              data-hf-link
              [attr.aria-label]="texts.link_label"
              [placeholder]="texts.link_placeholder"
              [value]="link()"
              (input)="update(link, $any($event.target).value)"
          /></label>
          @if (texts.file_label && !namesFile()) {
            <label
              >{{ texts.file_label
              }}<input
                data-hf-file
                [attr.aria-label]="texts.file_label"
                [placeholder]="texts.file_placeholder"
                [value]="file()"
                (input)="update(file, $any($event.target).value)"
            /></label>
          }
          @if (!namesRevision()) {
            <label
              >{{ texts.revision_label
              }}<input
                data-hf-revision
                [attr.aria-label]="texts.revision_label"
                [placeholder]="texts.revision_placeholder"
                [value]="revision()"
                (input)="update(revision, $any($event.target).value)"
            /></label>
          }
          @if (status(); as said) {
            <p class="status" role="status" data-hf-status>{{ said }}</p>
          }
          <p class="help">{{ texts.help }}</p>
          <div class="actions">
            <button type="button" mat-button (click)="open.set(false)">Cancel</button
            ><button
              type="button"
              mat-button
              data-hf-use
              [disabled]="!selection()"
              (click)="choose()"
            >
              {{ texts.action }}
            </button>
          </div>
        </section>
      }
    </ng-template>`,
  styleUrl: './hf_file_picker.scss',
})
export class HfFilePicker {
  /** The server's form of the side's runtime; the button is off without one. */
  readonly form = input<HuggingFaceForm | undefined>(undefined);
  /** The side's runtime, as the server checks the model for it. */
  readonly runtime = input('LiteRT-LM');
  /** The side, for the button's name. */
  readonly side = input('');
  readonly disabled = input(false);
  readonly selected = output<HfModel>();
  readonly open = signal(false);
  readonly link = signal('');
  readonly file = signal('');
  readonly revision = signal('');
  /** What the server says of the model named: whether it has it, and its size; or why not. */
  readonly status = signal('');
  private readonly overlay = viewChild(CdkConnectedOverlay);
  private readonly injector = inject(Injector);
  private readonly artifactsApi = inject(ArtifactsApi);
  /** The parts of a huggingface.co link; null for a repository id or text that is no link. */
  private readonly url = computed(() => {
    try {
      const url = new URL(this.link().trim());
      return url.hostname === 'huggingface.co' ? url.pathname.split('/').filter(Boolean) : null;
    } catch {
      return null;
    }
  });
  /** Whether the link names a file itself (/blob/ or /resolve/<revision>/<file>). */
  readonly namesFile = computed(() => {
    const parts = this.url();
    return !!parts && ['blob', 'resolve'].includes(parts[2]) && parts.length > 4;
  });
  /** Whether the link names a revision itself. */
  readonly namesRevision = computed(() => {
    const parts = this.url();
    return !!parts && ['blob', 'resolve', 'tree'].includes(parts[2]) && parts.length > 3;
  });
  /** What is sent: the server checks it; here only that a repository is named. */
  readonly selection = computed<HfModel | null>(() => {
    const link = this.link().trim();
    if (!link || (!this.url() && !/^[^/\s]+\/[^/\s]+$/.test(link))) {
      return null;
    }
    return {
      link,
      file: this.namesFile() ? '' : this.file().trim(),
      revision: this.namesRevision() ? '' : this.revision().trim(),
    };
  });

  constructor() {
    // The model named, once the fields rest, is checked; a newer one cancels the check before.
    effect((onCleanup) => {
      const selected = this.selection();
      const runtime = this.runtime();
      this.status.set('');
      if (!selected || !this.open()) {
        return;
      }
      const controller = new AbortController();
      const timer = setTimeout(() => {
        this.artifactsApi
          .checkHuggingFace({runtime, ...selected}, controller.signal)
          .then((checked) => this.status.set(checked.text))
          .catch((error: unknown) => {
            if (!controller.signal.aborted) {
              this.status.set(error instanceof Error ? error.message : String(error));
            }
          })
          .finally(() =>
            afterNextRender(() => this.overlay()?.overlayRef?.updatePosition(), {
              injector: this.injector,
            }),
          );
      }, CHECK_DELAY_MS);
      onCleanup(() => {
        clearTimeout(timer);
        controller.abort();
      });
    });
  }

  show(): void {
    this.link.set('');
    this.file.set('');
    this.revision.set('');
    this.open.set(true);
  }

  update(field: WritableSignal<string>, value: string): void {
    field.set(value);
    // The fields shown change with the link, and the connected panel's size with them.
    afterNextRender(() => this.overlay()?.overlayRef?.updatePosition(), {injector: this.injector});
  }

  choose(): void {
    const selected = this.selection();
    if (!selected) {
      return;
    }
    this.open.set(false);
    this.selected.emit(selected);
  }
}
