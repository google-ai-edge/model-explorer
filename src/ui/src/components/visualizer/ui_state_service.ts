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

import {Injectable, signal} from '@angular/core';

import {PaneState, VisualizerUiState} from './common/visualizer_ui_state';

/**
 * A service to manage UI state.
 */
@Injectable()
export class UiStateService {
  readonly curUiState = signal<VisualizerUiState>({
    paneStates: [this.createInitialPaneState()],
  });

  setDeepestExpandedGroupNodeIds(ids: string[], paneIndex: number) {
    this.curUiState.update((uiState) => {
      const paneState = uiState.paneStates[paneIndex];
      if (!paneState) {
        return uiState;
      }
      paneState.deepestExpandedGroupNodeIds = ids;
      return {...uiState};
    });
  }

  setSelectedNodeId(id: string, paneIndex: number) {
    this.curUiState.update((uiState) => {
      const paneState = uiState.paneStates[paneIndex];
      if (!paneState) {
        return uiState;
      }
      paneState.selectedNodeId = id;
      return {...uiState};
    });
  }

  setSelectedGraphId(
    graphId: string,
    collectionLabel: string,
    paneIndex: number,
  ) {
    this.curUiState.update((uiState) => {
      const paneState = uiState.paneStates[paneIndex];
      if (!paneState) {
        return uiState;
      }
      paneState.selectedGraphId = graphId;
      paneState.selectedCollectionLabel = collectionLabel;
      return {...uiState};
    });
  }

  setFlattenLayers(flatten: boolean, paneIndex: number) {
    this.curUiState.update((uiState) => {
      const paneState = uiState.paneStates[paneIndex];
      if (!paneState) {
        return uiState;
      }
      paneState.flattenLayers = flatten;
      return {...uiState};
    });
  }

  addPane() {
    this.curUiState.update((uiState) => {
      if (uiState.paneStates.length > 1) {
        uiState.paneStates = [uiState.paneStates[0]];
      }
      uiState.paneStates.push(this.createInitialPaneState());
      for (const paneState of uiState.paneStates) {
        paneState.widthFraction = 0.5;
      }
      return {...uiState};
    });
  }

  removePane(paneIndex: number) {
    this.curUiState.update((uiState) => {
      uiState.paneStates.splice(paneIndex, 1);
      if (uiState.paneStates.length === 1) {
        uiState.paneStates[0].widthFraction = 1;
      }
      return {...uiState};
    });
  }

  resizePane(leftWidthFraction: number) {
    this.curUiState.update((uiState) => {
      if (uiState.paneStates.length === 2) {
        uiState.paneStates[0].widthFraction = leftWidthFraction;
        uiState.paneStates[1].widthFraction = 1 - leftWidthFraction;
      }
      return {...uiState};
    });
  }

  swapPane() {
    this.curUiState.update((uiState) => {
      if (uiState.paneStates.length === 2) {
        uiState.paneStates = [uiState.paneStates[1], uiState.paneStates[0]];
      }
      return {...uiState};
    });
  }

  selectPane(paneIndex: number) {
    this.curUiState.update((uiState) => {
      for (let i = 0; i < uiState.paneStates.length; i++) {
        const paneState = uiState.paneStates[i];
        paneState.selected = i === paneIndex;
      }
      return {...uiState};
    });
  }

  reset() {
    this.curUiState.set({
      paneStates: [this.createInitialPaneState()],
    });
  }

  private createInitialPaneState(): PaneState {
    return {
      deepestExpandedGroupNodeIds: [],
      selectedNodeId: '',
      selectedGraphId: '',
      selectedCollectionLabel: '',
      widthFraction: 1,
      selected: false,
    };
  }
}
