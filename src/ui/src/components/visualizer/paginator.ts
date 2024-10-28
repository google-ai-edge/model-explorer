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
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  EventEmitter,
  Input,
  Output,
} from '@angular/core';

import {MatIconModule} from '@angular/material/icon';

/**
 * A paginator for navigating through pages.
 */
@Component({
  standalone: true,
  selector: 'paginator',
  imports: [CommonModule, MatIconModule],
  templateUrl: './paginator.ng.html',
  styleUrls: ['./paginator.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class Paginator {
  @Input({required: true}) pageSize!: number;
  @Input({required: true}) itemsCount!: number;
  @Output('change') readonly change = new EventEmitter<number>();

  private curPageIndex = 0;

  constructor(private readonly changeDetectorRef: ChangeDetectorRef) {}

  reset() {
    this.curPageIndex = 0;
    this.changeDetectorRef.markForCheck();
  }

  handleClickGoToPrevPage() {
    this.curPageIndex--;
    this.curPageIndex = Math.max(0, this.curPageIndex);
    this.change.emit(this.curPageIndex);
  }

  handleClickGoToNextPage() {
    this.curPageIndex++;
    this.curPageIndex = Math.min(
      Math.ceil(this.itemsCount / this.pageSize) - 1,
      this.curPageIndex,
    );
    this.change.emit(this.curPageIndex);
  }

  get disablePrevButton(): boolean {
    return this.curPageIndex === 0;
  }

  get disableNextButton(): boolean {
    return this.curPageIndex === Math.ceil(this.itemsCount / this.pageSize) - 1;
  }

  get curRangeText(): string {
    const from = Math.min(
      this.itemsCount,
      this.curPageIndex * this.pageSize + 1,
    );
    const to = Math.min(
      (this.curPageIndex + 1) * this.pageSize,
      this.itemsCount,
    );
    if (from === to) {
      return `${from}`;
    }
    return `${from} - ${to}`;
  }
}
