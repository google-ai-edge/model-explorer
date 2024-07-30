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
  defaultValue: 'Const,pseudo_const,pseudo_qconst,ReadVariableOp',
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
