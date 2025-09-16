/**
 * @license
 * Copyright 2025 The Model Explorer Authors. All Rights Reserved.
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

import {ApplicationRef, provideZoneChangeDetection} from '@angular/core';
import {createCustomElement} from '@angular/elements';
import {createApplication} from '@angular/platform-browser';
import 'zone.js';

import {provideAnimations} from '@angular/platform-browser/animations';
import {Wrapper} from './wrapper';

const MODEL_EXPLORER_VISUALIZER_TAG = 'model-explorer-visualizer';

// Create the wrapper element and register it as a custom element with the
// browser.
createApplication({
  providers: [provideZoneChangeDetection(), provideAnimations()],
}).then((appRef: ApplicationRef) => {
  if (!customElements.get(MODEL_EXPLORER_VISUALIZER_TAG)) {
    const constructor = createCustomElement(Wrapper, {
      injector: appRef.injector,
    });
    customElements.define(MODEL_EXPLORER_VISUALIZER_TAG, constructor);
  } else {
    console.log(
      `Custom element '${MODEL_EXPLORER_VISUALIZER_TAG}' already registered`,
    );
  }
});

// tslint:disable-next-line:no-any
(window as any)['modelExplorer'] = {};

// Re-export all the common types used by the visualizer.
//
// These are mainly for exporting enums which will have actual javascript code.
export * from '../components/visualizer/common/edge_overlays';
export * from '../components/visualizer/common/input_graph';
export * from '../components/visualizer/common/sync_navigation';
export * from '../components/visualizer/common/types';
export * from '../components/visualizer/common/visualizer_config';
export * from '../components/visualizer/common/visualizer_ui_state';
