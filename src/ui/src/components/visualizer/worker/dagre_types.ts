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

/**
 * The main dagre type.
 */
export declare interface Dagre {
  graphlib: {Graph: DagreGraphConstructor};
  layout(graph: DagreGraphInstance): void;
}

/** Constructor type for a dagre graph. */
export declare interface DagreGraphConstructor {
  new (): DagreGraphInstance;
}

/** Type for a dagre graph. */
export declare interface DagreGraphInstance {
  setGraph(cnofig: DagreGraphConfig): DagreGraphInstance;
  setDefaultEdgeLabel(
    callback:
      | string
      | ((v: string, w: string, name?: string) => string | DagreLabel),
  ): DagreGraphInstance;
  setNode(name: string, label: string | DagreLabel): DagreGraphInstance;
  setEdge(
    sourceId: string,
    targetId: string,
    value?: string | DagreLabel,
    name?: string,
  ): DagreGraphInstance;
  edges(): DagreEdge[];
  edge(edgeObj: DagreEdge): DagreGraphEdge;
}

/** Graph config options. */
export declare interface DagreGraphConfig {
  width?: number | undefined;
  height?: number | undefined;
  compound?: boolean | undefined;
  rankdir?: string | undefined;
  align?: string | undefined;
  nodesep?: number | undefined;
  edgesep?: number | undefined;
  ranksep?: number | undefined;
  marginx?: number | undefined;
  marginy?: number | undefined;
  acyclicer?: string | undefined;
  ranker?: string | undefined;
}

/** A label. */
export declare interface DagreLabel {
  // tslint:disable-next-line:no-any Allow arbitrary types.
  [key: string]: any;
}

/** An edge in dagre. */
export declare interface DagreEdge {
  v: string;
  w: string;
  name?: string | undefined;
}

/** An edge in dagre with detail data. */
export declare interface DagreGraphEdge {
  points: Array<{x: number; y: number}>;
  // tslint:disable-next-line:no-any Allow arbitrary types.
  [key: string]: any;
}
