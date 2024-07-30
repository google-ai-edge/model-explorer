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

import {Inject, Injectable} from '@angular/core';

import {INJECT_WINDOW} from '../inject';

/** Declare local extension for GA related types. */
export declare interface WindowWithGtag extends Window {
  gtag?: Function;
}

/**
 * Parameters for GA events.
 *
 * See: https://support.google.com/analytics/answer/10075209
 */
export interface EventParams {
  [paramType: string]: string | number | boolean;
}

/**
 * Specific event types sent into the system.
 */
export enum GaEventType {
  SHARE_CLICKED = 'share_clicked', // LEGACY from home_page.ts

  RPC_ERROR = 'rpc_error', // category: rpc method, label: error message.
  UI_ERROR = 'ui_error', // category: ui context, label: error message.
  FILE_EXTENSION = 'file_extension', // category: file extension.
  ADAPTER_ID = 'adapter_id', // category: adapter id.

  NODE_COUNT = 'node_count', // value: node count.
  CONVERSION_TIME = 'conversion_time', // value: elapsed time in seconds.
}

/**
 * Service for GA related tasks.
 */
@Injectable({providedIn: 'root'})
export class GaService {
  readonly gtag;

  // Always inject the window object, so it may be mocked in tests.
  constructor(@Inject(INJECT_WINDOW) private readonly window: WindowWithGtag) {
    this.gtag = this.window.gtag;
  }

  /**
   * Sends a new event to GA.
   *
   * "category" and "label" can be queried directly from the GA4 UI and are the
   * preferred method for contextualizing events.  The usage of these params for
   * each event is listed above in comments on the enum.
   */
  trackEvent(
    type: GaEventType,
    category = '',
    label?: string,
    description?: string,
    value?: number,
  ) {
    const path = this.getPath(this.window.location.href);
    // Set some defaults.
    label = label ?? path;
    description = description ?? type;
    value = value ?? 1.0;

    // Compose GA4 event.
    const ga4Event = {
      'event_category': category,
      'event_label': label,
      'description': description,
      'value': value,
      'page_path': path,
    };

    this.trackEventInternal(type, ga4Event);
  }

  trackNumeric(type: GaEventType, value: number) {
    this.trackEvent(type, '', undefined, undefined, value);
  }

  private getPath(url: string) {
    // "http://google.com/path/to/the/file.html" => "/path/to/the/file.html"
    return '/' + url.split('/').slice(3).join('/');
  }

  /**
   * Sends a new event to GA.
   *
   * Event parameters listed in the following page will be automatically
   * processed by GA and can be used directly in reports.
   * https://support.google.com/analytics/answer/9216061?hl=en
   *
   * Other custom parameters need to be set up in manually. See:
   * https://support.google.com/analytics/answer/10075209
   */
  private trackEventInternal(event: string, params?: EventParams) {
    if (this.gtag) {
      this.gtag('event', event, {...params});
    }
  }
}
