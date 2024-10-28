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

import {setLocationHref} from 'safevalues/dom';
import {IS_EXTERNAL} from '../common/flags';
import {INTERNAL_COLAB} from '../common/utils';

import {Injectable} from '@angular/core';

enum DirectiveName {
  RefreshPage = 'refreshPage',
}

declare interface DirectiveBase {
  name: DirectiveName;
}

declare interface RefreshPageDirective extends DirectiveBase {
  name: DirectiveName.RefreshPage;
  url: string;
}

type Directive = RefreshPageDirective;

/** A service for handling directives streaming from ME server. */
@Injectable({
  providedIn: 'root',
})
export class ServerDirectorService {
  init() {
    if (IS_EXTERNAL && !INTERNAL_COLAB) {
      // Listen to the streaming events (directives) from the following source
      // that the server has established.
      const eventSource = new EventSource('/apistream/server_director');
      eventSource.addEventListener('message', (e) => {
        if (!e.data) {
          return;
        }
        const directive = JSON.parse(e.data) as Directive;
        switch (directive.name) {
          // Refresh page with the given url.
          case DirectiveName.RefreshPage:
            setLocationHref(
              window.location,
              directive.url,
            );

            break;
          default:
            break;
        }
      });
    }
  }
}
