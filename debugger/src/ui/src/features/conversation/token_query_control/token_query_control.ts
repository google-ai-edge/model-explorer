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
import {OverlayModule} from '@angular/cdk/overlay';
import {Component, input, output, signal} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {
  OverlayDialog,
  OverlayPanel,
} from '../../../shared/overlay_dialog/overlay_dialog';
import {TOKEN_FIELD_HELP, TOKEN_QUERY_EXAMPLES} from '../token_metric_metadata';
import {compileQuery, queryFields} from '../token_query';
@Component({
  selector: 'token-query-control',
  imports: [
    OverlayDialog,
    OverlayPanel,
    OverlayModule,
    A11yModule,
    FormsModule,
    MatButtonModule,
    MatIconModule,
  ],
  template: ` <button
      mat-button
      cdkOverlayOrigin
      #origin="cdkOverlayOrigin"
      aria-label="Find condition"
      [attr.aria-expanded]="open()"
      (click)="show()"
    >
      <mat-icon>filter_alt</mat-icon><span>Find</span
      ><span class="query-label">{{
        formula() === 'NOT token_match' ? 'Token mismatch' : formula()
      }}</span>
    </button>
    <ng-template
      overlayDialog
      [origin]="origin"
      [open]="open()"
      placement="above-start"
      (closed)="close()"
    >
      <section overlayPanel class="query-panel" aria-label="Find condition">
        <header>
          <strong>Find condition</strong
          ><button mat-icon-button aria-label="Close find condition" (click)="close()">
            <mat-icon>close</mat-icon>
          </button>
        </header>
        <form (ngSubmit)="apply()">
          <label class="formula-label" for="token-find-formula">Formula</label
          ><input
            #formulaInput
            id="token-find-formula"
            name="formula"
            aria-label="Find formula"
            [(ngModel)]="draft"
            (input)="error.set('')"
            (compositionstart)="composing.set(true)"
            (compositionend)="composing.set(false)"
            maxlength="1000"
            autocomplete="off"
            spellcheck="false"
            cdkFocusInitial
          /><button
            type="button"
            mat-button
            [disabled]="draft.trim() === 'NOT token_match'"
            (click)="draft = 'NOT token_match'; error.set('')"
          >
            Reset to default
          </button>
          @if (error()) {
            <p role="alert">{{ error() }}</p>
          }
          <details>
            <summary>Formula reference</summary>
            <div class="formula-reference-body">
              <strong class="reference-heading">Fields · click to insert</strong>
              <div aria-label="Formula fields">
                @for (field of fields; track field) {
                  <button
                    type="button"
                    class="field-button"
                    mat-button
                    [disabled]="composing()"
                    [attr.aria-label]="'Insert ' + field"
                    (mousedown)="$event.preventDefault()"
                    (click)="insertField(field, formulaInput)"
                  >
                    <code>{{ field }}</code
                    ><span>{{ fieldHelp[field] }}</span>
                  </button>
                }
              </div>
              <strong class="reference-heading">Operators</strong>
              <dl>
                <dt>Compare</dt>
                <dd><code>&gt; &gt;= &lt; &lt;= == !=</code></dd>
                <dt>Combine</dt>
                <dd><code>AND · OR · NOT</code></dd>
                <dt>Group</dt>
                <dd><code>( ... )</code></dd>
              </dl>
              <strong class="reference-heading">Examples</strong>
              @for (example of examples; track example.formula) {
                <p class="formula-example">
                  <code>{{ example.formula }}</code
                  ><span>{{ example.description }}</span>
                </p>
              }
              <p>
                Only true conditions match. Unavailable values stay unknown, even with NOT.
                One-sided tokens have token_match = false; pair metrics are unavailable.
              </p>
            </div>
          </details>
          <footer><button mat-flat-button color="primary" type="submit">Apply</button></footer>
        </form>
      </section>
    </ng-template>`,
  styleUrl: './token_query_control.scss',
})
export class TokenQueryControl {
  readonly formula = input.required<string>();
  readonly formulaChange = output<string>();
  readonly open = signal(false);
  readonly error = signal('');
  readonly fields = Object.keys(queryFields);
  draft = '';
  readonly fieldHelp = TOKEN_FIELD_HELP;
  readonly examples = TOKEN_QUERY_EXAMPLES;
  readonly composing = signal(false);
  insertField(field: string, input: HTMLInputElement) {
    if (this.composing()) return;
    // Same caret/selection and field-prefix replacement as the candidates demo.
    const start = input.selectionStart ?? input.value.length,
      end = input.selectionEnd ?? start;
    const prefix = input.value.slice(0, start).match(/[A-Za-z_]+$/)?.[0] ?? '';
    input.setRangeText(field, start - prefix.length, end, 'end');
    this.draft = input.value;
    this.error.set('');
    input.focus({preventScroll: true});
  }
  show() {
    this.draft = this.formula();
    this.error.set('');
    this.composing.set(false);
    this.open.set(true);
  }
  close() {
    this.open.set(false);
  }
  apply() {
    try {
      compileQuery(this.draft);
      this.formulaChange.emit(this.draft);
      this.close();
    } catch (e) {
      this.error.set((e as Error).message);
    }
  }
}
