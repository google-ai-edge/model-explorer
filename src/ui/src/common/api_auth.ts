/**
 * @license
 * Copyright 2026 The Model Explorer Authors. All Rights Reserved.
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

import {IS_EXTERNAL} from './flags';

/**
 * The HTTP header that carries the access token of the Model Explorer server.
 *
 * The Python server requires the token on its API requests (see `auth.py`), and
 * only hands it out through the URL it prints, opens or embeds (`?token=...`).
 */
export const API_TOKEN_HEADER = 'X-Model-Explorer-Token';

/** The URL query parameter that carries the access token. */
export const API_TOKEN_QUERY_PARAM = 'token';

const API_TOKEN_SESSION_STORAGE_KEY = 'model_explorer_api_token';

/**
 * Why the server rejected the page's access token: the page sent no token, or
 * a token that isn't the server's current one (e.g. from an earlier run).
 */
export type ApiTokenRejectionReason = 'missing' | 'invalid';

const API_TOKEN_REJECTED_MESSAGE =
  'The Model Explorer server rejected this page: the URL must have a "token" ' +
  "parameter with the server's current access token. Open the full URL " +
  'printed by the server.';

/** Dependencies of this module, overridable in tests. */
export const apiAuthDeps = {
  /**
   * Whether the server API needs the access token. Only the external version
   * (served by the Python server) uses it, not the internal version.
   */
  tokenAuthEnabled: IS_EXTERNAL,

  getLocationHref: (): string => window.location.href,

  /** Whether the page is the custom element, which sets this global. */
  isCustomElement: (): boolean => Reflect.get(window, 'modelExplorer') != null,

  /**
   * Tells the user that the server rejected the page's token. The app shows
   * its dialog instead, through `registerApiTokenRejectedHandler`.
   */
  onApiTokenRejected: (reason: ApiTokenRejectionReason): void => {
    console.error(`${API_TOKEN_REJECTED_MESSAGE} (token ${reason})`);
  },
};

let reportedApiTokenRejection = false;

/**
 * Sets how to tell the user that the server rejected the page's access token
 * (e.g. with an Angular dialog).
 */
export function registerApiTokenRejectedHandler(
  handler: (reason: ApiTokenRejectionReason) => void,
): void {
  apiAuthDeps.onApiTokenRejected = handler;
}

/** Resets the state of this module. Only for tests. */
export function resetApiAuthForTesting(): void {
  reportedApiTokenRejection = false;
}

function readStoredApiToken(): string {
  try {
    return window.sessionStorage.getItem(API_TOKEN_SESSION_STORAGE_KEY) ?? '';
  } catch (e) {
    // Storage might be unavailable (e.g. in some sandboxed iframes).
    return '';
  }
}

function storeApiToken(token: string): void {
  try {
    window.sessionStorage.setItem(API_TOKEN_SESSION_STORAGE_KEY, token);
  } catch (e) {
    // Ignore. The token in the page URL still works.
  }
}

/**
 * Returns the access token of the Model Explorer server, or an empty string if
 * there is none (or the token is not used by this version).
 *
 * The token comes from the page URL. It is also kept in the session storage so
 * that it survives navigations that drop it from the URL.
 */
export function getApiToken(): string {
  if (!apiAuthDeps.tokenAuthEnabled) {
    return '';
  }
  const tokenFromUrl =
    new URL(apiAuthDeps.getLocationHref()).searchParams.get(
      API_TOKEN_QUERY_PARAM,
    ) ?? '';
  if (tokenFromUrl) {
    storeApiToken(tokenFromUrl);
    return tokenFromUrl;
  }
  return readStoredApiToken();
}

/**
 * Returns the given URL with the access token added as a query parameter.
 *
 * Only use it for requests that can't set headers (e.g. `EventSource`).
 */
export function addApiTokenToUrl(url: string): string {
  const token = getApiToken();
  if (!token) {
    return url;
  }
  const separator = url.includes('?') ? '&' : '?';
  const tokenParam = `${API_TOKEN_QUERY_PARAM}=${encodeURIComponent(token)}`;
  return `${url}${separator}${tokenParam}`;
}

/**
 * Tells the user (once per page load) that the server rejected the token.
 * Not in the custom element, whose host page owns the server connection.
 */
function reportApiTokenRejection(reason: ApiTokenRejectionReason): void {
  if (reportedApiTokenRejection || apiAuthDeps.isCustomElement()) {
    return;
  }
  reportedApiTokenRejection = true;
  apiAuthDeps.onApiTokenRejected(reason);
}

/** Returns a copy of the request options with the access token header. */
function withApiToken(
  init: RequestInit | undefined,
  token: string,
): RequestInit {
  const headers = new Headers(init?.headers);
  headers.set(API_TOKEN_HEADER, token);
  return {...init, headers};
}

/**
 * Sends a request to the Model Explorer server API.
 *
 * Same as `fetch`, except that it adds the access token of the server (if any)
 * and tells the user how to fix it when the server rejects the token.
 */
export async function apiFetch(
  url: string,
  init?: RequestInit,
): Promise<Response> {
  const token = getApiToken();
  const resp = await fetch(url, token ? withApiToken(init, token) : init);
  if (apiAuthDeps.tokenAuthEnabled && resp.status === 401) {
    reportApiTokenRejection(token ? 'invalid' : 'missing');
  }
  return resp;
}
