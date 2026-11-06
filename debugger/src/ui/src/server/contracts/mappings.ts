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

/** One side's tensor of a pair made by hand: an operator output of its signature's graph, and the
 *  point that output carries there (what is captured of it). */
export interface PairTensor {
  signature: string;
  node: string;
  output: string;
  point: string;
}

/** A pair made by hand, as a Session of its two models has it. */
export interface ManualPair {
  id: string;
  /** The user's name for it, or what its two tensors are. */
  label: string;
  created: string;
  ref: PairTensor;
  target: PairTensor;
  /** Where Intermediate Tensors has it (its rows' layer and anchor); null with `text` saying
   *  why. */
  place: {layer: number; anchor: string} | null;
  /** Why it has no place, or that a PyTorch module it names is captured from the next start only;
   *  '' otherwise. */
  text: string;
}

/** GET, POST and DELETE /api/mappings?session_id=: the pairs made by hand of a Session's two
 *  models. 'identity' for one model on both sides, which needs none; 'none' when none can be made;
 *  `text` says why. */
export interface ManualPairs {
  status: 'pairs' | 'identity' | 'none';
  text: string;
  pairs: ManualPair[];
}

/** What the editor posts to save a pair: each side's operator output. */
export interface SavePair {
  ref: Omit<PairTensor, 'point'>;
  target: Omit<PairTensor, 'point'>;
  label?: string;
}
