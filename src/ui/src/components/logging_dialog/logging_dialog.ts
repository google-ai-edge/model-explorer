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

import {CommonModule} from '@angular/common';
import {Component, Inject} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatDialogModule} from '@angular/material/dialog';
import {MatIconModule} from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';
import type { LoggingServiceInterface, LogLevel } from '../../common/logging_service_interface';

/**
 * A dialog showing app level settings.
 */
@Component({
  selector: 'logging-dialog',
  standalone: true,
  imports: [
    CommonModule,
    MatButtonModule,
    MatDialogModule,
    MatIconModule,
    MatTooltipModule,
  ],
  templateUrl: './logging_dialog.ng.html',
  styleUrls: ['./logging_dialog.scss'],
})
export class LoggingDialog {
  private dateFormatter = new Intl.DateTimeFormat('en', { hour: '2-digit', minute: '2-digit', second: '2-digit', fractionalSecondDigits: 3, hour12: false });
  constructor(
    @Inject('LoggingService')
    private readonly loggingService: LoggingServiceInterface,
  ) {}

  get messages() {
    return this.loggingService.getMessages();
  }

  getLogLevelIcon(level: LogLevel) {
    switch (level) {
      case 'info':
        return 'info';
      case 'warn':
        return 'warning';
      case 'error':
        return 'dangerous';
      case 'log':
      default:
        return 'help';
    }
  }

  clearLogs() {
    this.loggingService.clear();
  }

  downloadLogs() {
    const messages = this.loggingService.getMessages();

    if (messages.length > 0) {
      const tempElement = document.createElement('a');
      const textUrl = URL.createObjectURL(new Blob([JSON.stringify(messages, null, '\t')], { type: 'application/json' }));

      tempElement.hidden = true;
      tempElement.download = `logs-${new Date().toISOString()}.json`;
      tempElement.href = textUrl;
      tempElement.click();

      URL.revokeObjectURL(textUrl);
    }
  }

  formatDate(date: Date) {
    return this.dateFormatter.format(date);
  }
}
