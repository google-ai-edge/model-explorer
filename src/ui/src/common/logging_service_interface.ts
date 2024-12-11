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

/** Available log levels for the logger */
export const LogLevels = {
  error: 0,
  warn: 1,
  info: 2,
  log: 3,
  debug: 4
} as const;

export type LogLevel = keyof typeof LogLevels;

export interface LogMessage {
  timestamp: Date;
  level: LogLevel;
  messages: string[]
}

/** The interface of the logging service. */
export interface LoggingServiceInterface {
  currentlogLevel: LogLevel;
  log(...messages: string[]): void
  info(...messages: string[]): void
  warn(...messages: string[]): void
  error(...messages: string[]): void
  debug(...messages: string[]): void
  getMessages(level?: LogLevel): LogMessage[]
}
