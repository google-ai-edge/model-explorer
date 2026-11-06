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

/**
 * Chat-level (conversation) settings shared by the New chat form and the configuration overview.
 * They are the server's: it describes each one for a device (RuntimeCapability.chat_options,
 * option_catalog.py). Each chat of a Session runs with its own, set as it starts.
 */
import type {OptionDescription} from '../server/contracts/runtime';

export interface OptionGroup {
  name: string;
  keys: string[];
}

/** The settings in their sections, in the order the server describes them. */
export function chatOptionGroups(options: readonly OptionDescription[]): OptionGroup[] {
  const groups: OptionGroup[] = [];
  for (const option of options) {
    const group = groups.find((g) => g.name === option.group);
    if (group) {
      group.keys.push(option.key);
    } else {
      groups.push({name: option.group, keys: [option.key]});
    }
  }
  return groups;
}
export function switchLabel(value: unknown): string {
  return value === 'on' ? 'Enabled' : value === 'off' ? 'Disabled' : 'Model default';
}
