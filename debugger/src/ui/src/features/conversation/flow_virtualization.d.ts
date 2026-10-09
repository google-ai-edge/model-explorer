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

export interface FlowRow {
  index: number;
  ref: number | null;
  target: number | null;
}
export interface FlowTurn {
  steps: {ref: string[]; target: string[]};
  inputTokens: {ref: string[]; target: string[]};
  alignments: Partial<Record<string, FlowRow[]>>;
}
export type FlowAnchor =
  | {kind: 'absolute'; scrollTop: number}
  | {
      kind: 'token';
      ti: number;
      side: 'ref' | 'target';
      step: number;
      stage: 'Thinking' | 'Output';
      offset: number;
      windowLine?: number | null;
    }
  | {
      kind: 'input';
      ti: number;
      position: number;
      stage: 'Prefill';
      offset: number;
      windowLine?: number | null;
    };
export interface FlowConfig {
  threshold?: number;
  getTurn: (turn: number) => FlowTurn;
  getRows: (turn: number) => FlowRow[];
  getAlignment: () => string;
  isDebug: () => boolean;
  getScroller: () => HTMLElement | null;
  getVisibleBounds?: () => {top: number; bottom: number} | null;
  phase: (turn: number, side: string, index: number) => string;
  findRow: (turn: number, side: string, index: number) => FlowRow | undefined;
  getSelectedRow: (turn: number) => FlowRow | null;
  renderOutputToken: (
    word: string,
    index: number,
    side: string,
    turn: number,
  ) => string;
  renderInputToken: (
    word: string,
    index: number,
    side: string,
    turn: number,
  ) => string;
  displayKey: () => string;
  styleKey: () => string;
  widthAdjustment?: () => number;
  lineHeight?: () => number;
  onSelect: (turn: number, index: number, side: string) => void;
  afterScroll?: () => void;
}
export interface FlowVirtualizer {
  usesOutput: (turn: number) => boolean;
  usesInput: (turn: number) => boolean;
  mount: (anchor?: FlowAnchor | null) => void;
  paint: (force?: boolean, geometryOnly?: boolean) => void;
  jump: (turn: number, index: number, side?: string, pick?: boolean) => boolean;
  capture: (repaint?: boolean) => FlowAnchor | null;
  restore: (anchor: FlowAnchor | null) => void;
  clearLayout: () => void;
  destroy: () => void;
}
export function createFlowVirtualizer(config: FlowConfig): FlowVirtualizer;
