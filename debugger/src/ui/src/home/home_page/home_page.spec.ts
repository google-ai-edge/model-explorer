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

import {ApplicationRef} from '@angular/core';
import {TestBed} from '@angular/core/testing';
import {MatSnackBar} from '@angular/material/snack-bar';
import type {SessionSummary} from '../../server/contracts/sessions';
import {RuntimeService} from '../../server/runtime_service';
import type {UploadOptions} from '../../server/server_client';
import {SessionsApi} from '../../server/sessions_api';
import {
  allowed,
  chat,
  deferred,
  flushMicrotasks,
  listing,
  refused,
  session,
} from '../../server/testing/session_fixtures';
import {PreferencesService} from '../../state/preferences_service';
import {WorkspaceStateService} from '../../state/workspace_state_service';
import {HomePage} from './home_page';

/** A Session that has ended after a turn: its chat is read, and it may be exported. */
const ended = session('ended', {
  name: 'CPU vs GPU',
  status_text: 'Ended',
  has_capture: true,
  chats: [chat(1, {state: 'ended', has_capture: true})],
  model_server: {state: 'ended', text: 'Ended', error: '', button: null},
  actions: {...session().actions, enter: allowed, delete: allowed, export: allowed},
});

describe('HomePage', () => {
  let imported: ReturnType<typeof deferred<SessionSummary>>;
  let upload: UploadOptions;
  const snacks: string[] = [];
  const managed: Array<[string, unknown]> = [];

  beforeEach(() => {
    // The page is hidden: the list polls only when the spec asks it to.
    Object.defineProperty(document, 'visibilityState', {configurable: true, get: () => 'hidden'});
    imported = deferred<SessionSummary>();
    snacks.length = 0;
    managed.length = 0;
    TestBed.configureTestingModule({
      imports: [HomePage],
      providers: [
        {
          provide: SessionsApi,
          useValue: {
            sessions: async () => listing([session('starting'), ended]),
            exportUrl: (id: string) => `/api/sessions/${id}/export`,
            manageSession: async (operation: string, payload: {id: string}) => {
              managed.push([operation, payload]);
              return payload.id === 'ended' ? ended : session(payload.id);
            },
            importSession: (file: File, options: UploadOptions) => {
              upload = options;
              return imported.promise;
            },
          },
        },
        {
          provide: RuntimeService,
          useValue: {
            capabilities: () => null,
            lifecyclePending: () => ({}),
            lifecycleErrors: () => ({}),
          },
        },
        {provide: WorkspaceStateService, useValue: {open: () => {}}},
        {provide: PreferencesService, useValue: {showWelcome: () => false}},
        {
          provide: MatSnackBar,
          useValue: {
            open: (text: string) => {
              snacks.push(text);
              return {onAction: () => ({subscribe: () => {}})};
            },
          },
        },
      ],
    });
  });
  afterEach(() => {
    Reflect.deleteProperty(document, 'visibilityState');
    document.querySelectorAll('.cdk-overlay-container').forEach((el) => el.remove());
  });

  async function rendered() {
    // The editor is deferred: its metadata resolves first.
    await TestBed.compileComponents();
    const fixture = TestBed.createComponent(HomePage);
    fixture.detectChanges();
    await flushMicrotasks();
    fixture.detectChanges();
    return fixture;
  }

  it('Import, left of New session, uploads a .mdsession file with a progress row', async () => {
    const fixture = await rendered();
    const page = fixture.nativeElement as HTMLElement;
    const actions = Array.from(page.querySelectorAll('.header-actions button'));
    expect(actions.map((b) => b.textContent?.trim())).toEqual(['uploadImport', 'addNew session']);
    expect(actions[0].hasAttribute('mat-stroked-button')).toBe(true);
    const input = page.querySelector<HTMLInputElement>('input.import-input')!;
    expect(input.type).toBe('file');
    expect(input.accept).toBe('.mdsession');
    const picked = vi.spyOn(input, 'click').mockImplementation(() => {});
    (actions[0] as HTMLButtonElement).click();
    expect(picked).toHaveBeenCalled();

    const file = new File(['x'.repeat(100)], 'CPU vs GPU.mdsession');
    const chosen = {files: [file], value: 'C:\\fakepath\\CPU vs GPU.mdsession'};
    const importing = fixture.componentInstance.importFile(chosen as unknown as HTMLInputElement);
    expect(chosen.value, 'the same file may be chosen again').toBe('');
    upload.onProgress!(45, 100);
    fixture.detectChanges();
    const row = () => page.querySelector<HTMLElement>('tbody tr.import-row');
    expect(row()?.textContent).toContain('CPU vs GPU.mdsession');
    expect(row()?.textContent).toContain('Uploading · 45%');
    expect(page.querySelector<HTMLButtonElement>('.import-session')!.disabled).toBe(true);
    upload.onProgress!(100, 100);
    fixture.detectChanges();
    expect(row()?.textContent).toContain('Unpacking…');

    imported.resolve(session('copy', {...ended, id: 'copy', name: 'CPU vs GPU'}));
    await importing;
    fixture.detectChanges();
    expect(row()).toBeNull();
    const ids = Array.from(page.querySelectorAll('tbody tr[data-session-id]')).map((tr) =>
      tr.getAttribute('data-session-id'),
    );
    // Imported just now, it is the last active: first.
    expect(ids).toEqual(['copy', 'starting', 'ended']);
    expect(snacks).toEqual(['Imported CPU vs GPU. Open it to debug.']);
  });

  it('an import the server refuses says why and leaves no row', async () => {
    const fixture = await rendered();
    const file = new File(['x'], 'notes.mdsession');
    const importing = fixture.componentInstance.importFile({
      files: [file],
      value: '',
    } as unknown as HTMLInputElement);
    imported.reject(new Error('This is not a .mdsession file.'));
    await importing;
    fixture.detectChanges();
    expect((fixture.nativeElement as HTMLElement).querySelector('tr.import-row')).toBeNull();
    expect(snacks).toEqual(['Import failed: This is not a .mdsession file.']);
  });

  it('Delete asks first, says the captured data goes for good, then deletes with no undo', async () => {
    const fixture = await rendered();
    const page = fixture.nativeElement as HTMLElement;
    const app = TestBed.inject(ApplicationRef);
    const rows = () =>
      Array.from(page.querySelectorAll('tbody tr[data-session-id]')).map((tr) =>
        tr.getAttribute('data-session-id'),
      );
    const dialog = () => document.querySelector<HTMLElement>('session-delete');
    const dialogButton = (text: string) =>
      Array.from(dialog()!.querySelectorAll<HTMLButtonElement>('button')).find(
        (button) => button.textContent?.trim() === text,
      )!;
    async function openDelete() {
      page
        .querySelector<HTMLButtonElement>(
          'tr[data-session-id="ended"] [data-session-action="more"]',
        )!
        .click();
      fixture.detectChanges();
      const items = Array.from(
        document.querySelectorAll<HTMLButtonElement>('.mat-mdc-menu-panel button'),
      ).filter((item) => item.textContent?.includes('Delete'));
      items[items.length - 1].click();
      await vi.waitFor(() => expect(dialog()).not.toBeNull());
      app.tick();
    }
    async function closed() {
      await vi.waitFor(() => expect(dialog()).toBeNull());
      fixture.detectChanges();
    }

    await openDelete();
    const text = dialog()!.textContent!.replace(/\s+/g, ' ');
    expect(text).toContain('Delete session?');
    expect(text).toContain('Delete CPU vs GPU for good?');
    expect(text).toContain("This frees the session's captured data.");
    expect(text).toContain('It cannot be undone.');
    expect(text).not.toContain('undo this action');
    dialogButton('Cancel').click();
    await closed();
    expect(managed, 'nothing is deleted before it is confirmed').toEqual([]);
    expect(rows()).toEqual(['starting', 'ended']);

    await openDelete();
    dialogButton('Delete').click();
    await closed();
    expect(managed).toEqual([['delete', {id: 'ended'}]]);
    expect(rows()).toEqual(['starting']);
    expect(snacks, 'no undo is offered').toEqual([]);
  });

  it('Export in the More actions menu saves an ended Session; otherwise it says why', async () => {
    const fixture = await rendered();
    const page = fixture.nativeElement as HTMLElement;
    const clicked: string[] = [];
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (
      this: HTMLAnchorElement,
    ) {
      clicked.push(this.getAttribute('href') + ' ' + this.hasAttribute('download'));
    });
    const exportItem = (id: string) => {
      page
        .querySelector<HTMLButtonElement>(
          `tr[data-session-id="${id}"] [data-session-action="more"]`,
        )!
        .click();
      fixture.detectChanges();
      const items = document.querySelectorAll<HTMLButtonElement>('[data-session-action="export"]');
      return items[items.length - 1];
    };
    const starting = exportItem('starting');
    expect(starting.textContent?.trim()).toBe('downloadExport');
    expect(starting.disabled).toBe(true);
    expect(starting.getAttribute('aria-description')).toBe('This Session has no turn to export.');
    starting.click();
    expect(clicked).toEqual([]);

    const item = exportItem('ended');
    expect(item.disabled).toBe(false);
    item.click();
    expect(clicked).toEqual(['/api/sessions/ended/export true']);
    // Exporting… until the server has sent it all: no second Export, no Open meanwhile.
    fixture.detectChanges();
    const row = page.querySelector<HTMLElement>('tr[data-session-id="ended"]')!;
    expect(row.querySelector('.exporting')?.textContent?.trim()).toBe('Exporting…');
    const again = exportItem('ended');
    expect(again.disabled).toBe(true);
    expect(again.getAttribute('aria-description')).toBe(
      'This Session is being exported; wait until its file is saved.',
    );
    expect(row.querySelector<HTMLButtonElement>('[data-session-action="open"]')!.disabled).toBe(
      true,
    );
  });

  it('a Session the server lists as exporting says so, and waits', async () => {
    TestBed.overrideProvider(SessionsApi, {
      useValue: {
        sessions: async () => listing([{...ended, exporting: true}]),
        exportUrl: (id: string) => `/api/sessions/${id}/export`,
      },
    });
    const fixture = await rendered();
    const row = (fixture.nativeElement as HTMLElement).querySelector<HTMLElement>(
      'tr[data-session-id="ended"]',
    )!;
    expect(row.querySelector('.exporting')?.textContent?.trim()).toBe('Exporting…');
    expect(row.querySelector<HTMLButtonElement>('[data-session-action="open"]')!.disabled).toBe(
      true,
    );
  });

  it('a Session whose start failed reads Ended · no turn, with why, and is not opened', async () => {
    const failed = session('failed', {
      status_text: 'Ended · no turn',
      model_server: {
        state: 'ended',
        text: 'Ended · no turn',
        error: 'Reference: No Runner App is running on lzr-15.',
        button: null,
      },
      actions: {
        ...session().actions,
        enter: refused('This Session ended with no turn; to try again, create a new Session.'),
      },
    });
    TestBed.overrideProvider(SessionsApi, {
      useValue: {
        sessions: async () => listing([failed]),
        exportUrl: (id: string) => `/api/sessions/${id}/export`,
      },
    });
    const fixture = await rendered();
    const row = (fixture.nativeElement as HTMLElement).querySelector<HTMLElement>(
      'tr[data-session-id="failed"]',
    )!;
    const switchButton = row.querySelector<HTMLButtonElement>('.model-server')!;
    expect(switchButton.textContent).toContain('Ended · no turn');
    expect(switchButton.disabled).toBe(true);
    expect(row.querySelector('.server-error')?.textContent).toBe(
      'Reference: No Runner App is running on lzr-15.',
    );
    expect(row.querySelector<HTMLButtonElement>('[data-session-action="open"]')!.disabled).toBe(
      true,
    );
    expect(row.textContent).not.toContain('Try again');
    expect(row.textContent).not.toContain('Draft');
  });
});
