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

import {signal} from '@angular/core';

export const BOOKMARK_STORAGE_VERSION = 1;
export const BOOKMARK_SAVE_ERROR =
  'Bookmarks could not be saved in this browser.';

/** How one feature validates and identifies its bookmark entries. */
export interface BookmarkCodec<T> {
  /** The validated entry, or null to drop a corrupt one without losing its neighbours. */
  parse(value: unknown): T | null;
  /** Entries with the same identity are one bookmark; the first occurrence wins. */
  identity(value: T): string;
}
export interface BookmarkStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export function dedupeBookmarks<T>(
  items: readonly T[],
  codec: BookmarkCodec<T>,
): T[] {
  const seen = new Set<string>();
  return items.filter((item) => {
    const id = codec.identity(item);
    if (seen.has(id)) return false;
    seen.add(id);
    return true;
  });
}
/** Reads the versioned envelope `{version, items}` or the legacy bare array. */
export function decodeBookmarks<T>(
  raw: string | null,
  codec: BookmarkCodec<T>,
): T[] {
  let data: unknown;
  try {
    data = JSON.parse(raw ?? '[]');
  } catch {
    return [];
  }
  const entries: unknown[] = Array.isArray(data)
    ? data
    : data &&
        typeof data === 'object' &&
        Array.isArray((data as {items?: unknown}).items)
      ? (data as {items: unknown[]}).items
      : [];
  return dedupeBookmarks(
    entries
      .map((entry) => codec.parse(entry))
      .filter((item): item is T => item !== null),
    codec,
  );
}
export function encodeBookmarks<T>(items: readonly T[]): string {
  return JSON.stringify({version: BOOKMARK_STORAGE_VERSION, items});
}

/**
 * Bookmark lists keyed by storage key (per capture, per chat, or one per feature).
 * Replaced lists are cached in a signal so `computed()` readers update; a browser that
 * refuses the write (quota, private mode) is reported through `error`, never thrown.
 */
export class BookmarkStore<T> {
  private readonly lists = signal<Record<string, T[]>>({});
  readonly error = signal('');
  constructor(
    readonly codec: BookmarkCodec<T>,
    private readonly storage: () => BookmarkStorage = () => localStorage,
  ) {}
  read(key: string): T[] {
    const known = this.lists()[key];
    if (known) return known;
    try {
      return decodeBookmarks(this.storage().getItem(key), this.codec);
    } catch {
      return [];
    }
  }
  /** Replaces one list; returns false when the browser refused to persist it. */
  set(key: string, items: readonly T[]): boolean {
    const next = dedupeBookmarks(items, this.codec);
    this.lists.update((lists) => ({...lists, [key]: next}));
    try {
      this.storage().setItem(key, encodeBookmarks(next));
      this.error.set('');
      return true;
    } catch {
      this.error.set(BOOKMARK_SAVE_ERROR);
      return false;
    }
  }
}
