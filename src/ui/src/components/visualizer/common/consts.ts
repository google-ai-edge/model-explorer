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

import {IS_EXTERNAL} from '../../../common/flags';

/** The height of the node label. */
export const NODE_LABEL_HEIGHT = 11;

/**
 * The padding between the label and the value in node's attrs table.
 */
export const NODE_ATTRS_TABLE_LABEL_VALUE_PADDING = 4;

/** The padding of the left and right of the attrs table. */
export const NODE_ATTRS_TABLE_LEFT_RIGHT_PADDING = 4;

/** The font size of the attrs table. */
export const NODE_ATTRS_TABLE_FONT_SIZE = 9;

/** The top margin above node's attrs table. */
export const NODE_ATTRS_TABLE_MARGIN_TOP = 16;

/** The max width of the node's attrs table values. */
export const NODE_ATTRS_TABLE_VALUE_MAX_WIDTH = 200;

/** The height of attrs table row. */
export const NODE_ATTRS_TABLE_ROW_HEIGHT = 12;

/** The height of the summary row in node data provider. */
export const EXPANDED_NODE_DATA_PROVIDER_SUMMARY_ROW_HEIGHT = 14;

/** The top padding of the summary row in node data provider. */
export const EXPANDED_NODE_DATA_PROVIDER_SUMMARY_TOP_PADDING = 6;

/** The bottom padding of the summary row in node data provider. */
export const EXPANDED_NODE_DATA_PROVIDER_SUMMARY_BOTTOM_PADDING = 6;

/** The font size of the summary row in node data provider. */
export const EXPANDED_NODE_DATA_PROVIDER_SYUMMARY_FONT_SIZE = 9;

/** The maximum number of children nodes under a group node. */
export const DEFAULT_GROUP_NODE_CHILDREN_COUNT_THRESHOLD = IS_EXTERNAL
  ? 1000
  : 400;

/** The corner radius of the op node. */
export const OP_NODE_CORNER_RADIUS = 6;

/** Y factor for elements rendered in webgl. */
export const WEBGL_ELEMENT_Y_FACTOR = 0.001;

/** Number of segments on a curve. */
export const WEBGL_CURVE_SEGMENTS = 25;

/** Height of the bg color bars for node data provider. */
export const NODE_DATA_PROVIDER_BG_COLOR_BAR_HEIGHT = 5;

/** The key to expose test related objects. */
export const GLOBAL_KEY = 'me_test';

/** The tension for the catmullrom curve. */
export const CATMULLROM_CURVE_TENSION = 0.1;

/** The key to store the show on node item types in local storage. */
export const LOCAL_STORAGE_KEY_SHOW_ON_NODE_ITEM_TYPES =
  'model_explorer_show_on_node_item_types_v2';

/** The key to store the show on edge item types in local storage. */
export const LOCAL_STORAGE_KEY_SHOW_ON_EDGE_ITEM =
  'model_explorer_show_on_edge_item_v3';

/**
 * The key to store the show on edge item types in local storage
 * (old version).
 */
export const LOCAL_STORAGE_KEY_SHOW_ON_EDGE_ITEM_TYPES_V2 =
  'model_explorer_show_on_edge_item_types_v2';

/** The prefix for node data provider show on node type. */
export const NODE_DATA_PROVIDER_SHOW_ON_NODE_TYPE_PREFIX =
  'Node data provider: ';

/** The maximum number of input/output rows in the attrs table. */
export const MAX_IO_ROWS_IN_ATTRS_TABLE = 10;

/** The default font size of the edge label. */
export const DEFAULT_EDGE_LABEL_FONT_SIZE = 7.5;

/** The key to store the tensor values in node attributes. */
export const TENSOR_VALUES_KEY = '__value';

/** The key to store the tensor tag in i/o metadata. */
export const TENSOR_TAG_METADATA_KEY = '__tensor_tag';

/** The margin for the left and right side of the layout. */
export const LAYOUT_MARGIN_X = 20;

/** A map from color names to the corresponding hex color. */
export const COLOR_NAME_TO_HEX: Record<string, string> = {
  'aliceblue': '#f0f8ff',
  'antiquewhite': '#faebd7',
  'aqua': '#00ffff',
  'aquamarine': '#7fffd4',
  'azure': '#f0ffff',
  'beige': '#f5f5dc',
  'bisque': '#ffe4c4',
  'black': '#000000',
  'blanchedalmond': '#ffebcd',
  'blue': '#0000ff',
  'blueviolet': '#8a2be2',
  'brown': '#a52a2a',
  'burlywood': '#deb887',
  'cadetblue': '#5f9ea0',
  'chartreuse': '#7fff00',
  'chocolate': '#d2691e',
  'coral': '#ff7f50',
  'cornflowerblue': '#6495ed',
  'cornsilk': '#fff8dc',
  'crimson': '#dc143c',
  'cyan': '#00ffff',
  'darkblue': '#00008b',
  'darkcyan': '#008b8b',
  'darkgoldenrod': '#b8860b',
  'darkgray': '#a9a9a9',
  'darkgreen': '#006400',
  'darkkhaki': '#bdb76b',
  'darkmagenta': '#8b008b',
  'darkolivegreen': '#556b2f',
  'darkorange': '#ff8c00',
  'darkorchid': '#9932cc',
  'darkred': '#8b0000',
  'darksalmon': '#e9967a',
  'darkseagreen': '#8fbc8f',
  'darkslateblue': '#483d8b',
  'darkslategray': '#2f4f4f',
  'darkturquoise': '#00ced1',
  'darkviolet': '#9400d3',
  'deeppink': '#ff1493',
  'deepskyblue': '#00bfff',
  'dimgray': '#696969',
  'dodgerblue': '#1e90ff',
  'firebrick': '#b22222',
  'floralwhite': '#fffaf0',
  'forestgreen': '#228b22',
  'fuchsia': '#ff00ff',
  'gainsboro': '#dcdcdc',
  'ghostwhite': '#f8f8ff',
  'gold': '#ffd700',
  'goldenrod': '#daa520',
  'gray': '#808080',
  'green': '#008000',
  'greenyellow': '#adff2f',
  'honeydew': '#f0fff0',
  'hotpink': '#ff69b4',
  'indianred ': '#cd5c5c',
  'indigo': '#4b0082',
  'ivory': '#fffff0',
  'khaki': '#f0e68c',
  'lavender': '#e6e6fa',
  'lavenderblush': '#fff0f5',
  'lawngreen': '#7cfc00',
  'lemonchiffon': '#fffacd',
  'lightblue': '#add8e6',
  'lightcoral': '#f08080',
  'lightcyan': '#e0ffff',
  'lightgoldenrodyellow': '#fafad2',
  'lightgrey': '#d3d3d3',
  'lightgreen': '#90ee90',
  'lightpink': '#ffb6c1',
  'lightsalmon': '#ffa07a',
  'lightseagreen': '#20b2aa',
  'lightskyblue': '#87cefa',
  'lightslategray': '#778899',
  'lightsteelblue': '#b0c4de',
  'lightyellow': '#ffffe0',
  'lime': '#00ff00',
  'limegreen': '#32cd32',
  'linen': '#faf0e6',
  'magenta': '#ff00ff',
  'maroon': '#800000',
  'mediumaquamarine': '#66cdaa',
  'mediumblue': '#0000cd',
  'mediumorchid': '#ba55d3',
  'mediumpurple': '#9370d8',
  'mediumseagreen': '#3cb371',
  'mediumslateblue': '#7b68ee',
  'mediumspringgreen': '#00fa9a',
  'mediumturquoise': '#48d1cc',
  'mediumvioletred': '#c71585',
  'midnightblue': '#191970',
  'mintcream': '#f5fffa',
  'mistyrose': '#ffe4e1',
  'moccasin': '#ffe4b5',
  'navajowhite': '#ffdead',
  'navy': '#000080',
  'oldlace': '#fdf5e6',
  'olive': '#808000',
  'olivedrab': '#6b8e23',
  'orange': '#ffa500',
  'orangered': '#ff4500',
  'orchid': '#da70d6',
  'palegoldenrod': '#eee8aa',
  'palegreen': '#98fb98',
  'paleturquoise': '#afeeee',
  'palevioletred': '#d87093',
  'papayawhip': '#ffefd5',
  'peachpuff': '#ffdab9',
  'peru': '#cd853f',
  'pink': '#ffc0cb',
  'plum': '#dda0dd',
  'powderblue': '#b0e0e6',
  'purple': '#800080',
  'rebeccapurple': '#663399',
  'red': '#ff0000',
  'rosybrown': '#bc8f8f',
  'royalblue': '#4169e1',
  'saddlebrown': '#8b4513',
  'salmon': '#fa8072',
  'sandybrown': '#f4a460',
  'seagreen': '#2e8b57',
  'seashell': '#fff5ee',
  'sienna': '#a0522d',
  'silver': '#c0c0c0',
  'skyblue': '#87ceeb',
  'slateblue': '#6a5acd',
  'slategray': '#708090',
  'snow': '#fffafa',
  'springgreen': '#00ff7f',
  'steelblue': '#4682b4',
  'tan': '#d2b48c',
  'teal': '#008080',
  'thistle': '#d8bfd8',
  'tomato': '#ff6347',
  'turquoise': '#40e0d0',
  'violet': '#ee82ee',
  'wheat': '#f5deb3',
  'white': '#ffffff',
  'whitesmoke': '#f5f5f5',
  'yellow': '#ffff00',
  'yellowgreen': '#9acd32',
};

/** The port number for external local dev server. */
export const EXTERNAL_LOCAL_DEV_PORT = 8081;

/** The command to export to resource. */
export const EXPORT_TO_RESOURCE_CMD = 'model-explorer-export-to-resource';

/** The command to export selected nodes. */
export const EXPORT_SELECTED_NODES_CMD = 'model-explorer-export-selected-nodes';

/** The line height of node label. */
export const NODE_LABEL_LINE_HEIGHT = 14;
