/**
 * @license
 * Copyright 2024 The Model Explorer Authors. All Rights Reserved.
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

import {Component} from '@angular/core';

/** A component showing the app logo */
@Component({
  standalone: true,
  selector: 'me-logo',
  template: `<svg width="24" height="24" viewBox="0 0 256 256" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="256" height="256" fill="none"/>
  <path d="M229 64.5V192.5L128 248.5L127.5 115.5L229 64.5Z" [attr.fill]="fillUrl"/>
  <path d="M208 76V99L128 141.5V116L208 76Z" fill="white"/>
  <path d="M208 122.5V141.5L128 182.5V161.5L208 122.5Z" fill="white"/>
  <path d="M208 164.5V183L128 225.5V206L208 164.5Z" fill="white"/>
  <path d="M149 106V214.5L127 226V117.5L149 106Z" fill="white"/>
  <path d="M26 61L47.5 72.5V200L26 187V61Z" fill="#FBBC04"/>
  <path d="M125.5 10.5L145.5 21V60.5L125.5 56.5V10.5Z" fill="#DCA810"/>
  <path d="M164 31L184 41.5V81L164 78V31Z" fill="#309C4D"/>
  <path d="M164 31L184 41.5L87 94L65 82L164 31Z" fill="#11792D"/>
  <path d="M208 54.5L229 64.5L128 116L107 105L208 54.5Z" fill="#2C5CAC"/>
  <path d="M125.5 10.5L145.5 21L47.5 72.5L26 61L125.5 10.5Z" fill="#C78B15"/>
  <path d="M65 82L87 94V224L65 210.5V82Z" fill="#34A853"/>
  <path d="M107 105L128 116V248.5L107 236V105Z" fill="#4285F4"/>
  <defs>
  <linearGradient [attr.id]="linearGradientId" x1="144.5" y1="232" x2="219" y2="69.5" gradientUnits="userSpaceOnUse">
    <stop stop-color="#4285F4"/>
    <stop offset="1" stop-color="#2C5CAC"/>
  </linearGradient>
  </defs>
</svg>`,
  styles: `:host {display: flex; align-items: center; justify-content: center}`,
})
export class Logo {
  linearGradientId = Math.random().toString(36).slice(-6);
  fillUrl = `url(#${this.linearGradientId})`;
}
