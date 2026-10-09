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

import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  inject,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import {PreferencesService} from '../../../app/preferences_service';
@Component({
  selector: 'welcome-card',
  imports: [MatButtonModule, MatIconModule, MatTooltipModule],
  templateUrl: './welcome_card.ng.html',
  styleUrl: './welcome_card.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class WelcomeCard {
  readonly preferences = inject(PreferencesService);
  private readonly element = inject<ElementRef<HTMLElement>>(ElementRef);
  dismiss() {
    const home = this.element.nativeElement.closest('home-page');
    this.preferences.setShowWelcome(false);
    requestAnimationFrame(() =>
      home
        ?.querySelector<HTMLElement>('button.new-session, #editor-title')
        ?.focus(),
    );
  }
}
