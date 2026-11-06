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
 * A run's Session-level options are the server's: it describes each one, for each runtime and
 * device (RuntimeCapability.session_options, option_catalog.py). Every value is saved as text:
 * digits, an enum value, or '' / 'on' / 'off' for a switch.
 */
import type {OptionDescription, RuntimeCapability} from '../server/contracts/runtime';
import type {RuntimeOptions} from '../server/contracts/sessions';
export type RuntimeOptionKey = keyof RuntimeOptions;
/** The Session options a capability describes, in the order they are shown. */
export function sessionOptions(capability?: RuntimeCapability): readonly OptionDescription[] {
  return capability?.session_options ?? [];
}
/** Whether an option applies to a run: what it requires, such as the CPU backend, is chosen. */
export function optionApplies(option: OptionDescription, run: RuntimeOptions): boolean {
  return !option.requires || run[option.requires.key as RuntimeOptionKey] === option.requires.value;
}
/**
 * A run's option as the configuration shows it: its value, the server's default, or '—' where
 * it does not apply.
 */
export function runtimeOptionValue(
  run: RuntimeOptions,
  key: RuntimeOptionKey,
  option: OptionDescription,
): string {
  if (!optionApplies(option, run)) {
    return '—';
  }
  const value = run[key];
  if (option.type === 'switch') {
    return value === 'on' ? 'On' : value === 'off' ? 'Off' : 'Default';
  }
  return String(value || (option.default ?? 'Default'));
}
