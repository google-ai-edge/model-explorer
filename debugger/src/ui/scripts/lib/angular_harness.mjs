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

// Shared harness for the node regression scripts: real Angular signals, effects
// and injection, with only browser transport/time replaced by the caller.
import '@angular/compiler';
import {
  createEnvironmentInjector,
  runInInjectionContext,
  ɵEffectScheduler as EffectScheduler,
  ɵChangeDetectionScheduler as ChangeDetectionScheduler,
} from '@angular/core';
import {build} from 'esbuild';
import {mkdir, mkdtemp} from 'node:fs/promises';
import {rmSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

/** src/ui */
export const root = fileURLToPath(new URL('../..', import.meta.url));

export function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return {promise, resolve, reject};
}

/** Lets queued microtasks (resolved requests, signal notifications) run. */
export async function flush(rounds = 8) {
  for (let i = 0; i < rounds; i++) await Promise.resolve();
}

/** A scratch directory under node_modules/.cache, removed when the process exits. */
export async function cacheDir(prefix) {
  const cache = path.join(root, 'node_modules/.cache');
  await mkdir(cache, {recursive: true});
  const dir = await mkdtemp(path.join(cache, prefix));
  const cleanup = () => rmSync(dir, {recursive: true, force: true});
  process.once('exit', cleanup);
  return {dir, cleanup};
}

/**
 * Bundle TypeScript sources against the installed Angular packages and import them.
 * `exports` maps an export name to a source path relative to src/ui; the key `*`
 * re-exports a whole module.
 */
export async function bundleSources(exports, prefix = 'harness-') {
  const {dir, cleanup} = await cacheDir(prefix);
  const contents = Object.entries(exports)
    .map(([name, source]) =>
      name === '*' ? `export * from './${source}';` : `export {${name}} from './${source}';`,
    )
    .join('\n');
  const outfile = path.join(dir, 'bundle.mjs');
  await build({
    stdin: {contents, resolveDir: root, loader: 'ts'},
    bundle: true,
    packages: 'external',
    platform: 'node',
    format: 'esm',
    outfile,
    logLevel: 'silent',
  });
  return {module: await import(pathToFileURL(outfile)), cleanup};
}

/**
 * An environment injector with Angular's real root effect scheduler. `run` constructs
 * services in injection context; `settle` runs the effects queued so far; `flush`
 * alternates microtasks and effects.
 */
export function angularContext(providers = []) {
  const scheduler = EffectScheduler.ɵprov.factory();
  const injector = createEnvironmentInjector([
    ...providers,
    {provide: EffectScheduler, useValue: scheduler},
    {provide: ChangeDetectionScheduler, useValue: {notify() {}}},
  ]);
  return {
    injector,
    run: (factory) => runInInjectionContext(injector, factory),
    settle: () => scheduler.flush(),
    flush: async (rounds = 8) => {
      for (let i = 0; i < rounds; i++) {
        await Promise.resolve();
        scheduler.flush();
      }
    },
    destroy: () => injector.destroy(),
  };
}

/** Temporarily replace globals (fetch, localStorage, timers…) for one block. */
export async function withGlobals(overrides, block) {
  const previous = new Map(Object.keys(overrides).map((key) => [key, globalThis[key]]));
  Object.assign(globalThis, overrides);
  try {
    return await block();
  } finally {
    for (const [key, value] of previous)
      if (value === undefined) delete globalThis[key];
      else globalThis[key] = value;
  }
}

/** Records every value a writable signal is set to, for assertions on transient states. */
export function recordHistory(signalRef) {
  const history = [];
  const set = signalRef.set;
  signalRef.set = (value) => {
    history.push(value);
    set(value);
  };
  signalRef.update = (fn) => signalRef.set(fn(signalRef()));
  signalRef.history = history;
  return history;
}
