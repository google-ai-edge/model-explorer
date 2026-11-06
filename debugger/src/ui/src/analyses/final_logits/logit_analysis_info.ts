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

import type {TokenRangeSideMetadata} from '../../server/contracts/token_analysis';
import type {ExplorerInfoItem} from '../../widgets/explorer_info_list/explorer_info_list';

/** The server's distinct values as Metadata writes them, one not reported as such. */
function valuesText(values: readonly (string | number | null)[]): string {
  return (
    values.map((value) => (value === null ? 'Not reported' : String(value))).join(' · ') ||
    'Not reported'
  );
}

/** The one value the server lists, or Mixed with each where it lists several. */
function uniformValue(values: readonly (string | number | null)[]): string {
  if (values.every((value) => value === null)) {
    return '—';
  }
  return values.length === 1 ? String(values[0]) : `Mixed (${valuesText(values)})`;
}

/** Logical input geometry, like KV metadata: one captured vocabulary vector per position, as the
 *  server describes the range's vectors of one side. */
export function logitSourceItems(metadata: TokenRangeSideMetadata): ExplorerInfoItem[] {
  const count = metadata.count ?? 0;
  const sizes = metadata.vocabulary_sizes;
  return [
    {
      label: 'Dimensions',
      value: '[Token position, Vocabulary]',
      segments: [
        {text: '['},
        {
          text: 'Token position',
          help: {
            title: 'Token position',
            rows: [
              {label: 'Count', value: String(count)},
              {
                label: 'Counting',
                value: 'Captured logits vectors; alignment gaps and unavailable inputs excluded',
              },
            ],
          },
        },
        {text: ', '},
        {
          text: 'Vocabulary',
          help: {
            title: 'Vocabulary',
            rows: [
              {label: 'Dimension', value: 'Vocabulary token IDs in each captured logits vector'},
              {label: 'Size', value: valuesText(sizes)},
              {label: 'Identity', value: valuesText(metadata.vocabulary_identities ?? [])},
            ],
          },
        },
        {text: ']'},
      ],
    },
    {label: 'Shape', value: count ? `[${count}, ${uniformValue(sizes)}]` : '—', expandable: true},
    {label: 'Dtype', value: uniformValue(metadata.dtypes ?? []), expandable: true},
  ];
}
