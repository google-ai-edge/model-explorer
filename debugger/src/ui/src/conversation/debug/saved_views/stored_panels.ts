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

import {keptStagePanels, restoreLogitPanel, type StageAnalysisPanel} from '../../stage_analysis';

/**
 * Where this browser keeps each chat's analysis panels, by its capture (ConversationState
 * .contextKey).
 */
export const STORED_PANELS_KEY = 'debugger.analysisPanels.v1';
/** How many chats' panels are kept: the most recently changed ones. */
export const STORED_PANELS_SESSIONS = 32;

/** What the browser keeps of one chat: its panels, and when they last changed. */
interface StoredSession {
  at: number;
  panels: StageAnalysisPanel[];
}
/** The browser's storage, as PreferencesService reads and writes it. */
export interface PanelStorage {
  read(key: string): string | null;
  save(key: string, value: string): void;
}

function readAll(storage: PanelStorage): Record<string, StoredSession> {
  try {
    const value: unknown = JSON.parse(storage.read(STORED_PANELS_KEY) ?? '{}');
    return value && typeof value === 'object' && !Array.isArray(value)
      ? (value as Record<string, StoredSession>)
      : {};
  } catch {
    return {};
  }
}

/**
 * A chat's analysis panels as this browser kept them, each valid one; null when none were
 * kept.
 */
export function readStoredPanels(
  storage: PanelStorage,
  session: string,
): StageAnalysisPanel[] | null {
  if (!session) {
    return null;
  }
  const kept = keptStagePanels(readAll(storage)[session]?.panels);
  return kept ? kept.map(restoreLogitPanel) : null;
}

/**
 * Keeps a chat's analysis panels in this browser, with those of the most recently changed
 * other chats (STORED_PANELS_SESSIONS in all).
 */
export function storePanels(
  storage: PanelStorage,
  session: string,
  panels: StageAnalysisPanel[],
  now = Date.now(),
): void {
  if (!session) {
    return;
  }
  const all = readAll(storage);
  if (JSON.stringify(all[session]?.panels) === JSON.stringify(panels)) {
    return;
  }
  all[session] = {at: now, panels};
  const kept = Object.entries(all)
    .filter(([, entry]) => entry && typeof entry.at === 'number')
    .sort(([, a], [, b]) => b.at - a.at)
    .slice(0, STORED_PANELS_SESSIONS);
  storage.save(STORED_PANELS_KEY, JSON.stringify(Object.fromEntries(kept)));
}
