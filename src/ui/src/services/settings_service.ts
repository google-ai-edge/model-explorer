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

import {Injectable} from '@angular/core';

import {
  DEFAULT_EDGE_LABEL_FONT_SIZE,
  DEFAULT_GROUP_NODE_CHILDREN_COUNT_THRESHOLD,
} from '../components/visualizer/common/consts';
import {LocalStorageService} from '../components/visualizer/local_storage_service';

/** Keys for all settings. */
export enum SettingKey {
  CONST_ELEMENT_COUNT_LIMIT = 'const_element_count_limit',
  SHOW_WELCOME_CARD = 'show_welcome_card',
  HIDE_OP_NODES_WITH_LABELS = 'hide_op_nodes_with_labels',
  ARTIFICIAL_LAYER_NODE_COUNT_THRESHOLD = 'artificial_layer_node_count_threshold',
  EDGE_LABEL_FONT_SIZE = 'edge_label_font_size',
  EDGE_COLOR = 'edge_color',
  DISALLOW_VERTICAL_EDGE_LABELS = 'disallow_vertical_edge_labels',
  KEEP_LAYERS_WITH_A_SINGLE_CHILD = 'keep_layers_with_a_single_child',
  SHOW_OP_NODE_OUT_OF_LAYER_EDGES_WITHOUT_SELECTING = 'show_op_node_out_of_layer_edges_without_selecting',
  HIGHLIGHT_LAYER_NODE_INPUTS_OUTPUTS = 'highlight_layer_node_inputs_outputs',
  HIDE_EMPTY_NODE_DATA_ENTRIES = 'hide_empty_node_data_entries',
  SHOW_SIDE_PANEL_ON_NODE_SELECTION = 'show_side_panel_on_node_selection',
}

/** Setting types. */
export enum SettingType {
  BOOLEAN,
  NUMBER,
  TEXT_MULTILINE,
  COLOR,
}

/** Interface of a setting. */
export declare interface Setting {
  label: string;
  key: SettingKey;
  type: SettingType;
  defaultValue?: boolean | number | string;
  help?: string;
}

/** Interface for saved settings in local storage. */
export declare interface SavedSettings {
  [key: string]: boolean | number | string;
}

/** Setting for max const element count. */
export const SETTING_MAX_CONST_ELEMENT_COUNT_LIMIT: Setting = {
  label: 'Maximum element count for constant tensor values',
  key: SettingKey.CONST_ELEMENT_COUNT_LIMIT,
  type: SettingType.NUMBER,
  defaultValue: 16,
  help:
    'Controls the number of values extracted from the constant tensors ' +
    'during model processing. Increasing this number may impact performance ' +
    'due to larger payload sizes.',
};

/** Setting for showing welcome card. */
export const SETTING_SHOW_WELCOME_CARD: Setting = {
  label: 'Show welcome card',
  key: SettingKey.SHOW_WELCOME_CARD,
  type: SettingType.BOOLEAN,
  defaultValue: true,
};

/** Setting for hiding op nodes by label. */
export const SETTING_HIDE_OP_NODES_WITH_LABELS: Setting = {
  label: 'Hide op nodes with labels below (comma separated)',
  key: SettingKey.HIDE_OP_NODES_WITH_LABELS,
  type: SettingType.TEXT_MULTILINE,
  defaultValue: 'Const,no_value,pseudo_const,pseudo_qconst,ReadVariableOp',
  help:
    'Removes op nodes from model graphs if their label matches any ' +
    'of the labels entered below.',
};

/** Setting for maximum number of nodes in an artificial layer. */
export const SETTING_ARTIFACIAL_LAYER_NODE_COUNT_THRESHOLD: Setting = {
  label: 'Maximum number of nodes in an artificial layer',
  key: SettingKey.ARTIFICIAL_LAYER_NODE_COUNT_THRESHOLD,
  type: SettingType.NUMBER,
  defaultValue: DEFAULT_GROUP_NODE_CHILDREN_COUNT_THRESHOLD,
  help:
    'Controls the maximum number of immediate child nodes displayed ' +
    'under a layer. When the number of child nodes exceeds this limit, ' +
    'Model Explorer automatically groups them into smaller, more manageable ' +
    'artificial layers to improve layout performance and readability.',
};

/** Setting for edge label font size. */
export const SETTING_EDGE_LABEL_FONT_SIZE: Setting = {
  label: 'Edge label font size',
  key: SettingKey.EDGE_LABEL_FONT_SIZE,
  type: SettingType.NUMBER,
  defaultValue: DEFAULT_EDGE_LABEL_FONT_SIZE,
};

/** Setting for edge color. */
export const SETTING_EDGE_COLOR: Setting = {
  label: 'Edge color',
  key: SettingKey.EDGE_COLOR,
  type: SettingType.COLOR,
  defaultValue: '#aaaaaa',
};

/** Setting for disabllowing laying out edge labels vertically. */
export const SETTING_DISALLOW_VERTICAL_EDGE_LABELS: Setting = {
  label: 'Disallow vertical edge labels',
  key: SettingKey.DISALLOW_VERTICAL_EDGE_LABELS,
  type: SettingType.BOOLEAN,
  defaultValue: false,
  // The actual help content is in ng.html.
  help: '-',
};

/** Setting for keeping layers with a single child. */
export const SETTING_KEEP_LAYERS_WITH_A_SINGLE_CHILD: Setting = {
  label: 'Keep layers with a single op node child',
  key: SettingKey.KEEP_LAYERS_WITH_A_SINGLE_CHILD,
  type: SettingType.BOOLEAN,
  defaultValue: false,
  help:
    'By default, layers with a single op node as its child are automatically ' +
    'removed to improve graph readability. ' +
    'Turn this toggle on to keep those layers.',
};

/** Setting for showing op node out-of-layer edges without selecting. */
export const SETTING_SHOW_OP_NODE_OUT_OF_LAYER_EDGES_WITHOUT_SELECTING: Setting =
  {
    label: 'Show op node out-of-layer edges without selecting',
    key: SettingKey.SHOW_OP_NODE_OUT_OF_LAYER_EDGES_WITHOUT_SELECTING,
    type: SettingType.BOOLEAN,
    defaultValue: false,
    help:
      "By default, an op node's edges that go out of the layer is only " +
      'visible when the op node is selected. Turn this toggle on to see ' +
      'those edges without needing to select the node. ⚠️ This feature will ' +
      'make the model graph look more noisy and harder to read ' +
      'especially for larger models.',
  };

/** Setting for highlighting layer node inputs and outputs. */
export const SETTING_HIGHLIGHT_LAYER_NODE_INPUTS_OUTPUTS: Setting = {
  label: 'Highlight inputs and outputs of the selected layer node',
  key: SettingKey.HIGHLIGHT_LAYER_NODE_INPUTS_OUTPUTS,
  type: SettingType.BOOLEAN,
  defaultValue: false,
  help:
    'By default, inputs and outputs are highlighted only when an op node ' +
    'is selected. Enable this setting to see inputs and outputs for a ' +
    'selected layer node, including all its descendant op nodes within ' +
    'that layer.',
};

/** Settings for hiding empty noda data entries. */
export const SETTING_HIDE_EMPTY_NODE_DATA_ENTRIES: Setting = {
  label: 'Hide node data entries with empty values',
  key: SettingKey.HIDE_EMPTY_NODE_DATA_ENTRIES,
  type: SettingType.BOOLEAN,
  defaultValue: false,
  help:
    'Enable this setting to hide node data entries ' +
    '(on node overlay and in side panel) with empty values.',
};

/** Setting for showing side panel on node selection. */
export const SETTING_SHOW_SIDE_PANEL_ON_NODE_SELECTION: Setting = {
  label: 'Show side panel only when a node is selected',
  key: SettingKey.SHOW_SIDE_PANEL_ON_NODE_SELECTION,
  type: SettingType.BOOLEAN,
  defaultValue: false,
};

const SETTINGS_LOCAL_STORAGE_KEY = 'model_explorer_settings';

/** All settings. */
export const ALL_SETTINGS = [
  SETTING_MAX_CONST_ELEMENT_COUNT_LIMIT,
  SETTING_HIDE_OP_NODES_WITH_LABELS,
  SETTING_ARTIFACIAL_LAYER_NODE_COUNT_THRESHOLD,
  SETTING_EDGE_LABEL_FONT_SIZE,
  SETTING_EDGE_COLOR,
  SETTING_KEEP_LAYERS_WITH_A_SINGLE_CHILD,
  SETTING_SHOW_WELCOME_CARD,
  SETTING_DISALLOW_VERTICAL_EDGE_LABELS,
  SETTING_SHOW_OP_NODE_OUT_OF_LAYER_EDGES_WITHOUT_SELECTING,
  SETTING_HIGHLIGHT_LAYER_NODE_INPUTS_OUTPUTS,
  SETTING_HIDE_EMPTY_NODE_DATA_ENTRIES,
  SETTING_SHOW_SIDE_PANEL_ON_NODE_SELECTION,
];

/**
 * Service for managing app settings.
 */
@Injectable({providedIn: 'root'})
export class SettingsService {
  private readonly savedSettings: SavedSettings;

  constructor(private readonly localStorageService: LocalStorageService) {
    // Load saved settings from local storage.
    const strSavedSettings =
      this.localStorageService.getItem(SETTINGS_LOCAL_STORAGE_KEY) || '';
    this.savedSettings =
      strSavedSettings === ''
        ? {}
        : (JSON.parse(strSavedSettings) as SavedSettings);
  }

  getBooleanValue(setting: Setting): boolean {
    if (this.savedSettings[setting.key] == null) {
      return setting.defaultValue === true;
    }
    return this.savedSettings[setting.key] === true;
  }

  getNumberValue(setting: Setting): number {
    const savedStrNumber = this.savedSettings[setting.key];
    if (savedStrNumber != null) {
      return Number(savedStrNumber);
    }
    return (setting.defaultValue as number) || 0;
  }

  getStringValue(setting: Setting): string {
    const savedStrString = this.savedSettings[setting.key] as string;
    if (savedStrString != null) {
      return savedStrString;
    }
    return (setting.defaultValue as string) || '';
  }

  saveBooleanValue(value: boolean, settingKey: SettingKey) {
    this.savedSettings[settingKey] = value;
    this.localStorageService.setItem(
      SETTINGS_LOCAL_STORAGE_KEY,
      JSON.stringify(this.savedSettings),
    );
  }

  saveNumberValue(value: number, settingKey: SettingKey) {
    if (isNaN(value)) {
      return;
    }

    this.savedSettings[settingKey] = value;
    this.localStorageService.setItem(
      SETTINGS_LOCAL_STORAGE_KEY,
      JSON.stringify(this.savedSettings),
    );
  }

  saveStringValue(value: string, settingKey: SettingKey) {
    this.savedSettings[settingKey] = value;
    this.localStorageService.setItem(
      SETTINGS_LOCAL_STORAGE_KEY,
      JSON.stringify(this.savedSettings),
    );
  }

  getAllSettingsValues(): SavedSettings {
    const settingsValues: SavedSettings = {};
    for (const setting of ALL_SETTINGS) {
      switch (setting.type) {
        case SettingType.BOOLEAN:
          settingsValues[setting.key] = this.getBooleanValue(setting);
          break;
        case SettingType.NUMBER:
          settingsValues[setting.key] = this.getNumberValue(setting);
          break;
        default:
          break;
      }
    }
    return settingsValues;
  }

  getSettingByKey(settingKey: SettingKey): Setting | undefined {
    return ALL_SETTINGS.find((setting) => setting.key === settingKey);
  }
}
