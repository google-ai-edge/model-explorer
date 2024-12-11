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

import { Injectable } from '@angular/core';
import { LoggingServiceInterface, LogMessage, LogLevel, LogLevels } from '../common/logging_service_interface';

/**
 * A service to manage model loading related tasks.
 */
@Injectable({
  providedIn: 'root',
})
export class LoggingService implements LoggingServiceInterface {

  currentlogLevel = (localStorage.getItem('logLevel') as LogLevel) ?? 'log';

  readonly messages: LogMessage[] = [];

  constructor() {}

  private addLogMessages(level: LogLevel, ...messages: string[]) {
    this.messages.push({
        timestamp: new Date(),
        level,
        messages
      });
  }

  log(...messages: string[]): void {
    this.addLogMessages('log', ...messages);
  }

  info(...messages: string[]): void {
    this.addLogMessages('info', ...messages);
  }

  warn(...messages: string[]): void {
    this.addLogMessages('warn', ...messages);
  }

  error(...messages: string[]): void {
    this.addLogMessages('error', ...messages);
  }

  debug(...messages: string[]): void {
    this.addLogMessages('debug', ...messages);
  }

  getMessages(level?: LogLevel): LogMessage[] {
    if (!level) {
      return this.messages.filter(({ level: curLevel }) => LogLevels[curLevel] <= LogLevels[this.currentlogLevel]);
    }

    return this.messages.filter(({ level: curLevel }) => curLevel === level);
  }
}
