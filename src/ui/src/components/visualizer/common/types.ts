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

import {GroupNode, ModelGraph, ModelNode} from './model_graph';
import {SyncNavigationMode} from './sync_navigation';

/** A type for key-value pairs. */
export type KeyValuePairs = Record<string, string>;

/** An object with "key" and "value" field. */
export declare interface KeyValue {
  key: string;
  value: string;
}

/** A type for a list of key-value pairs. */
export type KeyValueList = KeyValue[];

/** A list of node attributes. */
export type NodeAttributeList = NodeAttribute[];

/** Node attributes as a record. */
export type NodeAttributePairs = Record<string, NodeAttributeValue>;

/** A type for a single node attribute. */
export declare interface NodeAttribute {
  key: string;
  value: NodeAttributeValue;
}

/** A single node attribute value. */
export type NodeAttributeValue = string | SpecialNodeAttributeValue;

/** non-string node attribute value. */
export type SpecialNodeAttributeValue =
  | NodeIdsNodeAttributeValue
  | NodesWithAttributeValues;

/** Node attribute value types. */
export enum NodeAttributeValueType {
  NODE_IDS = 'node_ids',
  NODE_WITH_ATTRS = 'node_with_attrs',
}

/**
 * A "node ids" node attribute value.
 *
 * Clicking on a node id will jump to the corresponding node in the graph.
 */
export declare interface NodeIdsNodeAttributeValue {
  type: NodeAttributeValueType.NODE_IDS;
  nodeIds: string[];
}

/**
 * A "node with attrs".
 *
 * Clicking on a node id will jump to the corresponding node in the graph.
 */
export declare interface NodesWithAttributeValues {
  type: NodeAttributeValueType.NODE_WITH_ATTRS;
  nodes: {
    id: string;
    attrs: KeyValueList;
  }[];
}

/** An item in input/output metadata. */
export interface MetadataItem {
  id: string;
  attrs: KeyValueList;
}

/** An incoming edge in the graph. */
export declare interface IncomingEdge {
  /** The id of the source node (where the edge comes from). */
  sourceNodeId: string;

  /** The id of the output from the source node that this edge goes out of. */
  sourceNodeOutputId: string;

  /**
   * The id of the input from the target node (this node) that this edge
   * connects to.
   */
  targetNodeInputId: string;

  /** Other associated metadata for this edge. */
  metadata?: KeyValuePairs;
}

/** An outgoing edge in the graph. */
export declare interface OutgoingEdge {
  /** The id of the target node (where the edge connects to). */
  targetNodeId: string;

  /** The id of the output from the source node that this edge goes out of. */
  sourceNodeOutputId: string;

  /**
   * The id of the input from the target node (this node) that this edge
   * connects to.
   */
  targetNodeInputId: string;

  /** Other associated metadata for this edge. */
  metadata?: KeyValuePairs;
}

/** A point with x and y coordinate. */
export declare interface Point {
  x: number;
  y: number;
}

/** A rectangle. */
export declare interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Attributes for group nodes. */
export declare interface GroupNodeAttributes {
  /**
   * From group's namespace (including the name of the group itself) to its
   * attribuets (key-value pairs).
   *
   * Use empty group namespace for the model-level attributes (i.e. shown in
   * side panel when no node is selected).
   */
  [namespaceName: string]: Record<string, GroupNodeAttributeItem>;
}

/** A single attribute item for group node. */
export type GroupNodeAttributeItem = string;

/** Custom configs for group nodes matched by the namespace regex. */
export declare interface GroupNodeConfig {
  /**
   * The regex of the namespace of the group node. The namespace to match should
   * include the name of the group itself.
   *
   * For example, for a group/layer a->b->c, the namespace string that the regex
   * will try to match is "a/b/c"
   */
  namespaceRegex: string;

  /** The layout direction of the matching group nodes. */
  layoutDirection?: LayoutDirection;
}

/** Layout direction for group nodes. */
export enum LayoutDirection {
  TOP_BOTTOM,
  LEFT_RIGHT,
}

/** The style of the op node. */
export declare interface GraphNodeStyle {
  /**
   * The background color of the node.
   *
   * It should be in css format.
   */
  backgroundColor?: string;

  /**
   * The border color of the node.
   *
   * It should be in css format.
   */
  borderColor?: string;

  /**
   * The border color of the node when it is hovered.
   *
   * It should be in css format.
   */
  hoveredBorderColor?: string;

  /**
   * The text color of the node.
   *
   * It should be in css format.
   */
  textColor?: string;

  /**
   * The border width of the node.
   *
   * It should be a number.
   *
   * If not set, the default border width is 1.2
   */
  borderWidth?: number;
}

/** Custom configs for a graph node. */
export declare interface GraphNodeConfig {
  /** Whether to pin the node to the top of the group it belongs to. */
  pinToGroupTop?: boolean;
}

/** Data to pass along when clicking "open in popup" on a group node. */
export interface PopupPanelData {
  id: string;
  groupNode: GroupNode;
  initialPosition: Point;
  curModelGraph: ModelGraph;
}

/** Data to pass along when a node is located. */
export interface NodeLocatedData {
  nodeId: string;
  deepestExpandedGroupNodeIds: string[];
}

/** Data to pass along when revealing a node. */
export interface NodeToRevealInfo {
  graphId: string;
  paneId: string;
  nodeId: string;
}

/** Info about a renderer. */
export interface RendererInfo {
  id: string;
  ownerType: RendererOwner;
}

/** The owner of the renderer. */
export enum RendererOwner {
  MAIN_PANEL,
  POPUP,
  GRAPH_PANEL,
}

/** Info about a selected node. */
export interface SelectedNodeInfo {
  nodeId: string;
  rendererId: string;
  isGroupNode: boolean;
  noNodeShake?: boolean;
  triggerNavigationSync?: boolean;
}

/** Info about a node to locate. */
export interface LocateNodeInfo extends SelectedNodeInfo {
  select?: boolean;
}

/** Types of renderer. */
export enum RendererType {
  WEBGL,
}

/** Info to pass along when clicking "add snapshot" */
export interface AddSnapshotInfo {
  rendererId: string;
}

/** Info for restoring a snapshot. */
export interface RestoreSnapshotInfo {
  rendererId: string;
  snapshot: SnapshotData;
}

/** Data related to a snapshot. */
export interface SnapshotData {
  id: string;
  rect: Rect;
  imageBitmap: ImageBitmap;
  deepestExpandedGroupNodeIds?: string[];
  selectedNodeId?: string;
  showOnNodeItemTypes?: Record<string, ShowOnNodeItemData>;
  showOnEdgeItem?: ShowOnEdgeItemData;
  flattenLayers?: boolean;
}

/** Info to pass along when clicking "expand/collapse all layers" */
export interface ExpandOrCollapseAllGraphLayersInfo {
  expandOrCollapse: boolean;
  rendererId: string;
}

/** Info to pass along when clicking "download as png" */
export interface DownloadAsPngInfo {
  rendererId: string;
  // false: graph in viewport.
  fullGraph: boolean;
  // Whether to set background to transparent.
  transparentBackground: boolean;
}

/** The basic info of a node data provider run. */
export declare interface NodeDataProviderRunInfo {
  runId: string;
  runName: string;
}

/** Info of a node when hovered or double clicked. */
export declare interface NodeInfo {
  nodeId: string;
  graphId: string;
  collectionLabel: string;
  node?: ModelNode;
  paneId?: string;
}

/** Supported aggregated stats. */
export type AggregatedStat = 'min' | 'max' | 'sum' | 'avg';

/** Node data provider data for a single graph. */
export declare interface NodeDataProviderGraphData {
  /**
   * The name of the node data.
   *
   * The node data's name (mainly used for display purposes) is determined by
   * the following sources, in order of priority:
   *
   * 1) an explicitly specified name from this field, which overrides any other
   *    source;
   * 2) the `name` parameter in the API call, if applicable;
   * 3) the JSON file name, if the data is loaded from a JSON file.
   */
  name?: string;

  /**
   * Node data indexed by node keys.
   *
   * The key could be:
   * - Any of the output tensor names of a node.
   * - The node id specified in the input graph json (see input_graph.ts).
   */
  results: Record<string, NodeDataProviderResultData>;

  /**
   * Thresholds that define various ranges and the corresponding node styles
   * (e.g. node bg color) to be applied for that range.
   *
   * This is only used when `NodeDataProviderResultData.value` is a number.
   *
   * Take the following thresholds as an example:
   *
   * [
   *   {value: 10, bgColor: 'red'}
   *   {value: 50, bgColor: 'blue'}
   *   {value: 100, bgColor: 'yellow'}
   * ]
   *
   * This means:
   * - Node data with value <=10 have "red" background color.
   * - Node data with value >10 and <=50 have "blue" background color.
   * - Node data with value >50 and <=100 have "yellow" background color.
   * - Node data with value >100 have no background color (white).
   */
  thresholds?: ThresholdItem[];

  /**
   * A gradient that defines the stops (from 0 to 1) and the associated colors.
   * A stop value 0 correspondg to the minimum value in `results`, and a stop
   * value 1 corresponds to the maximum value in results. Stops for 0 and 1
   * should always be provided.
   *
   * When color-coding a node, the system uses the node's data value to
   * calculate a corresponding stop and interpolates between gradient colors.
   *
   * This field takes precedence over the `thresholds` field above.
   */
  gradient?: GradientItem[];

  /**
   * Whether to hide the corresponding column in aggregated stats table
   * (the first table).
   *
   * If all columns in that table are hidden, the whole table will be hidden.
   */
  hideInAggregatedStatsTable?: boolean;

  /**
   * Whether to hide the corresponding column in children stats table
   * (the second table).
   *
   * If all columns in that table are hidden, the whole table will be hidden.
   */
  hideInChildrenStatsTable?: boolean;

  /**
   * The stats to hide in the aggregated stats table (the first table).
   *
   * The value for the hidden stat will be displayed as '-'.
   */
  hideAggregatedStats?: AggregatedStat[];

  /**
   * Controls whether to display a detailed value distribution summary on the
   * group node.
   *
   * By default, a color bar representing the value distribution of
   * all descendant nodes is shown at the bottom of the group node. If this
   * field is set to true, we will show a more detailed summary, with each
   * value's label, percentage, and count shown on a separate line.
   *
   * For now this only works with non-numerical (e.g. string) node data values.
   */
  showExpandedSummaryOnGroupNode?: boolean;

  /**
   * Whether to display the label count columns in the children stats table in
   * the side panel.
   *
   * For now this only works with non-numerical (e.g. string) node data values.
   */
  showLabelCountColumnsInChildrenStatsTable?: boolean;
}

/** The top level node data provider data, indexed by graph id. */
export declare interface NodeDataProviderData {
  [key: string]: NodeDataProviderGraphData;
}

/** The data for a node data provider run. */
export declare interface NodeDataProviderRunData {
  runId: string;
  runName: string;
  extensionId: string;
  collectionId: string;
  remotePath?: string;

  nodeDataProviderData?: NodeDataProviderData;

  // selected: boolean;

  done: boolean;
  /** A number from 0 to 1 as progress. */
  progress?: number;

  // graphId -> {nodeId -> processed results}
  results?: Record<string, Record<string, NodeDataProviderResultProcessedData>>;
  error?: string;
}

/** Info for a value in a node data provider run. */
export declare interface NodeDataProviderValueInfo {
  label: string;
  bgColor: string;
  count: number;
}

/** The result data for a node in a node data provider run. */
export declare interface NodeDataProviderResultData {
  /** The original value of the result. */
  // tslint:disable-next-line:no-any Allow arbitrary types.
  value: any;

  /**
   * The bg color to render the corresponding node for.
   *
   * This value overrides the value calculated from the thresholds
   * (`NodeDataProviderData.thresholds`) if specified.
   */
  bgColor?: string;

  /**
   * The text color to render the corresponding node for.
   *
   * This value overrides the value calculated from the thresholds
   * (`NodeDataProviderData.thresholds`) if specified.
   */
  textColor?: string;
}

/** The processed result data for a node in a node data provider run. */
export declare interface NodeDataProviderResultProcessedData
  extends NodeDataProviderResultData {
  /**
   * The accumulated values from all the results whose key maps to this node.
   */
  // tslint:disable-next-line:no-any Allow arbitrary types.
  allValues: {[key: string]: any};

  /** The string representation of the value. */
  strValue: string;
}

/**
 * A threshold item with the upperbound value and its corresponding bg color.
 */
export declare interface ThresholdItem {
  value: number;
  bgColor: string;
  textColor?: string;
}

/** A gradient item with the stop and its corresponding colors. */
export declare interface GradientItem {
  /** A number from 0 to 1. */
  stop: number;
  /** Only support hex form (e.g. #aabb00) or color name (e.g. red) */
  bgColor?: string;
  /** Only support hex form (e.g. #aabb00) or color name (e.g. red) */
  textColor?: string;
}

/** A pane in the main UI. */
export interface Pane {
  id: string;
  widthFraction: number;
  selectedNodeInfo?: SelectedNodeInfo;
  hasArtificialLayers?: boolean;
  // Use this to reveal a node in this pane and select it.
  nodeIdToReveal?: string;
  // Graph id to snapshots.
  snapshots?: Record<string, SnapshotData[]>;
  // Whether to flatten all layers (ignore namespaces in node).
  flattenLayers?: boolean;
  snapshotToRestore?: SnapshotData;
  subgraphBreadcrumbs?: SubgraphBreadcrumbItem[];
  searchResults?: SearchResults;
  selectedNodeDataProviderRunId?: string;

  // Renderer id -> <item type shown on node -> data>
  showOnNodeItemTypes?: Record<string, Record<string, ShowOnNodeItemData>>;

  // Renderer id -> data
  showOnEdgeItems?: Record<string, ShowOnEdgeItemData>;

  modelGraph?: ModelGraph;
}

/** The data for processed model graph and related info. */
export interface ProcessedModelGraphData {
  modelGraph: ModelGraph;
  paneId: string;
}

/** Color used in various webgl components. Each field is from 0 to 1. */
export interface WebglColor {
  r: number;
  g: number;
  b: number;
}

/** A value for a shader uniform. */
export declare interface UniformValue {
  // tslint:disable-next-line:no-any Allow arbitrary types.
  value: any;
}

/** An event indicating a model graph has been processed. */
export interface ModelGraphProcessedEvent {
  modelGraph: ModelGraph;
  paneIndex: number;
}

/** A single item in subgraph breadcrumbs. */
export interface SubgraphBreadcrumbItem {
  graphId: string;
  snapshot?: SnapshotData;
}

/** Base interface for a search match. */
export interface SearchMatchBase {
  type: SearchMatchType;
}

/** Interface for a search match for a node label. */
export interface SearchMatchNodeLabel extends SearchMatchBase {
  type: SearchMatchType.NODE_LABEL;
}

/** Interface for a search match for an input metadata. */
export interface SearchMatchInputMetadata extends SearchMatchBase {
  type: SearchMatchType.INPUT_METADATA;
  matchedText: string;
}

/** Interface for a search match for an output metadata. */
export interface SearchMatchOutputMetadata extends SearchMatchBase {
  type: SearchMatchType.OUTPUT_METADATA;
  matchedText: string;
}

/** Interface for a search match for an attribute. */
export interface SearchMatchAttr extends SearchMatchBase {
  type: SearchMatchType.ATTRIBUTE;
  matchedAttrId: string;
}

/** Union type for search match. */
export type SearchMatch =
  | SearchMatchNodeLabel
  | SearchMatchInputMetadata
  | SearchMatchOutputMetadata
  | SearchMatchAttr;

/** Multiple search matches. */
export interface SearchMatches {
  matches: SearchMatch[];
  matchTypes: Set<string>;
}

/**
 * Types of search match.
 *
 * The value should correspond to a material icon.
 */
export enum SearchMatchType {
  NODE_LABEL = 'title',
  INPUT_METADATA = 'input',
  OUTPUT_METADATA = 'output',
  ATTRIBUTE = 'list',
}

/** Search results. */
export interface SearchResults {
  /** From node id to matches */
  results: Record<string, SearchMatch[]>;
}

/**
 * Item types to be shown on node.
 *
 * To add a entry, follow the steps below:
 *
 * 1. Add an entry here.
 * 2. Add to the ALL_SHOW_ON_NODE_ITEM_TYPES list in renderer_wrapper.ts.
 * 3. Update renderAttrsTable in webgl_renderer.ts.
 * 4. Update getNodeHeight and getNodeWidth in graph_layout.ts.
 */
export enum ShowOnNodeItemType {
  OP_NODE_ID = 'Op node id',
  OP_ATTRS = 'Op node attributes',
  OP_INPUTS = 'Op node inputs',
  OP_OUTPUTS = 'Op node outputs',
  LAYER_NODE_CHILDREN_COUNT = 'Layer node children count',
  LAYER_NODE_DESCENDANTS_COUNT = 'Layer node descendants count',
  LAYER_NODE_ATTRS = 'Layer node attributes',
}

/** Item types to be shown on edge. */
export enum ShowOnEdgeItemType {
  OFF = 'Off',
  TENSOR_SHAPE = 'Tensor shape',
  OUTPUT_METADATA = 'Output metadata',
  INPUT_METADATA = 'Input metadata',
  SOURCE_NODE_ATTR = 'Source node attribute',
  TARGET_NODE_ATTR = 'Target node attribute',
}

/** Weight of the font. */
export enum FontWeight {
  REGULAR,
  MEDIUM,
  BOLD,
  MONOSPACE_MEDIUM,
  ICONS,
}

/** Info for a char rendered in webgl. */
export declare interface CharInfo {
  char: string;
  width: number;
  height: number;
  xoffset: number;
  yoffset: number;
  xadvance: number;
  x: number;
  y: number;
}

/** Field labels in info panel. */
export enum FieldLabel {
  OP_NODE_ID = 'id',
  NUMBER_OF_CHILDREN = '#children',
  NUMBER_OF_DESCENDANTS = '#descendants',
}

/** Data for show on node item. */
export declare interface ShowOnNodeItemData {
  selected: boolean;
  filterRegex?: string;
}

/** Data for show on edge item. */
export declare interface ShowOnEdgeItemData {
  type: string;
  filterText?: string;
  outputMetadataKey?: string;
  inputMetadataKey?: string;
  sourceNodeAttrKey?: string;
  targetNodeAttrKey?: string;
}

/** Old data for show on edge item. */
export declare interface ShowOnEdgeItemOldData {
  selected: boolean;
}

/** The ids of the node style. */
export enum NodeStyleId {
  NODE_BG_COLOR = 'node_bg_color',
  NODE_TEXT_COLOR = 'node_text_color',
  NODE_BORDER_COLOR = 'node_border_color',
}

/** A rule for node styler. All fields should be serializable. */
export declare interface NodeStylerRule {
  /**
   * Quries are connected with AND.
   */
  queries: NodeQuery[];

  /**
   * The type of node to match.
   *
   * @deprecated The new version the of rule stores node type as a query in
   * `queries` above.
   */
  nodeType?: SearchNodeType;

  /**
   * Styles applied to the matched nodes.
   *
   * Indexed by style ids.
   */
  styles: Partial<Record<NodeStyleId, SerializedStyle | string>>;

  /**
   * Should set this to V2.
   */
  version?: NodeStylerRuleVersion;
}

/** Version of node styler rule. */
export enum NodeStylerRuleVersion {
  V2 = 'v2',
}

declare interface NodeQueryBase {
  type: NodeQueryType;
}

/** A rule width processed node styler rules. */
export interface ProcessedNodeStylerRule {
  queries: ProcessedNodeQuery[];
  nodeType?: SearchNodeType;
  styles: Record<string, SerializedStyle | string>;
}

declare interface ProcessedNodeQueryBase {
  type: NodeQueryType;
  matchTypes: Set<SearchMatchType>;
}

/** A node regex query. */
export declare interface NodeRegexQuery extends NodeQueryBase {
  type: NodeQueryType.REGEX;
  queryRegex: string;
  matchTypes: SearchMatchType[];
}

/** A processed node regex query (not serializable). */
export interface ProcessedNodeRegexQuery extends ProcessedNodeQueryBase {
  type: NodeQueryType.REGEX;
  queryRegex: RegExp;
}

/** A node attr value range query. */
export declare interface NodeAttrValueRangeQuery extends NodeQueryBase {
  type: NodeQueryType.ATTR_VALUE_RANGE;
  attrName: string;
  min: number;
  max: number;
}

/** A node type query. */
export declare interface NodeTypeQuery extends NodeQueryBase {
  type: NodeQueryType.NODE_TYPE;
  nodeType: SearchNodeType;
}

/** Union type for node query. */
export type NodeQuery =
  | NodeRegexQuery
  | NodeAttrValueRangeQuery
  | NodeTypeQuery;

/** Union type for processed node query. */
export type ProcessedNodeQuery =
  | ProcessedNodeRegexQuery
  | NodeAttrValueRangeQuery
  | NodeTypeQuery;

/** Types of node query. */
export enum NodeQueryType {
  REGEX = 'regex',
  ATTR_VALUE_RANGE = 'attr_value_range',
  NODE_TYPE = 'node_type',
}

/** Types of node to match in a search. */
export enum SearchNodeType {
  OP_NODES = 'op_nodes',
  LAYER_NODES = 'layer_nodes',
  OP_AND_LAYER_NODES = 'op_and_layer_nodes',
}

/** Serialized style. */
export declare interface SerializedStyle {
  id: NodeStyleId;
  value: string;
}

/** Response from reading a file. */
export declare interface ReadFileResp {
  content: string;
}

/** Event for sync navigation mode change. */
export declare interface SyncNavigationModeChangedEvent {
  mode: SyncNavigationMode;
  // Used when mode is LOAD_MAPPING_FROM_CNS.
  cnsPath?: string;
}

/** Config for legend panel. */
export declare interface LegendConfig {
  /** Whether to hide the "Op node" legend row. */
  hideOp?: boolean;

  /** Whether to hide the "Layer node" legend row. */
  hideLayer?: boolean;

  /** Whether to hide the "Artificial layer" legend row. */
  hideArtificialLayers?: boolean;

  /** Whether to hide the "Selected op" legend row. */
  hideSelectedOp?: boolean;

  /** Whether to hide the "Selected layer" legend row. */
  hideSelectedLayer?: boolean;

  /** Whether to hide the "Identical layer" legend row. */
  hideIdenticalLayers?: boolean;

  /** Whether to hide the "Inputs" legend row. */
  hideInputs?: boolean;

  /** Whether to hide the "Outputs" legend row. */
  hideOutputs?: boolean;

  /** Whether to hide the navigation shortcuts legend rows. */
  hideShortcuts?: boolean;

  /**
   * If set, rename the "op" label to this string.
   */
  renameOpTo?: string;

  /**
   * If set, rename the "layer" label to this string.
   */
  renameLayerTo?: string;

  /**
   * If set, rename the "inputs" label to this string.
   */
  renameInputsTo?: string;

  /**
   * If set, rename the "outputs" label to this string.
   */
  renameOutputsTo?: string;
}

/** Config for "View on node" menu. */
export declare interface ViewOnNodeConfig {
  /** Whether to hide the "Op node id" item. */
  hideOpNodeId?: boolean;

  /** Whether to hide the "Op node attributes" item. */
  hideOpNodeAttributes?: boolean;

  /** Whether to hide the "Op node inputs" item. */
  hideOpNodeInputs?: boolean;

  /** Whether to hide the "Op node outputs" item. */
  hideOpNodeOutputs?: boolean;

  /** Whether to hide the "Layer node children count" item. */
  hideLayerNodeChildrenCount?: boolean;

  /** Whether to hide the "Layer node descendants count" item. */
  hideLayerNodeDescendantsCount?: boolean;

  /** Whether to hide the "Layer node attributes" item. */
  hideLayerNodeAttributes?: boolean;

  /** Whether to hide the "View on edges" section. */
  hideViewOnEdgesSection?: boolean;

  /**
   * If set, rename the "Op node id" item to this string.
   */
  renameOpNodeIdTo?: string;

  /**
   * If set, rename the "Op node attributes" item to this string.
   */
  renameOpNodeAttributesTo?: string;

  /**
   * If set, rename the "Op node inputs" item to this string.
   */
  renameOpNodeInputsTo?: string;

  /**
   * If set, rename the "Op node outputs" item to this string.
   */
  renameOpNodeOutputsTo?: string;
}

/** Config for the toolbar. */
export declare interface ToolbarConfig {
  /** Whether to hide the "Expand/collapse all layers" button. */
  hideExpandCollapseAllLayers?: boolean;

  /** Whether to hide the "Flatten all layers" button. */
  hideFlattenAllLayers?: boolean;

  /** Whether to hide the "Custom edge overlays" button. */
  hideCustomEdgeOverlays?: boolean;
}

/** Base interface for a command. */
export declare interface CommandBase {
  type: CommandType;
  paneIndex: number;
}

/** Command to collapse the info panel in the given pane. */
export declare interface CollapseInfoPanelCommand extends CommandBase {
  type: CommandType.COLLAPSE_INFO_PANEL;
}

/** Command to show the info panel in the given pane. */
export declare interface ShowInfoPanelCommand extends CommandBase {
  type: CommandType.SHOW_INFO_PANEL;
}

/** Command to set the view on edge mode in the given pane. */
export declare interface SetViewOnEdgeCommand extends CommandBase {
  type: CommandType.SET_VIEW_ON_EDGE;
  mode: ViewOnEdgeMode;
  value?: string;
}

/** Modes for "View on edge". */
export enum ViewOnEdgeMode {
  OFF = 'off',
  TENSOR_SHAPE = 'tensor_shape',
  SOURCE_NODE_ATTR = 'source_node_attr',
  TARGET_NODE_ATTR = 'target_node_attr',
  OUTPUT_METADATA = 'output_metadata',
  INPUT_METADATA = 'input_metadata',
}

/** Command types. */
export enum CommandType {
  COLLAPSE_INFO_PANEL = 'collapse_info_panel',
  SHOW_INFO_PANEL = 'show_info_panel',
  SET_VIEW_ON_EDGE = 'set_view_on_edge',
}

/** Union type for all commands. */
export type Command =
  | CollapseInfoPanelCommand
  | ShowInfoPanelCommand
  | SetViewOnEdgeCommand;
