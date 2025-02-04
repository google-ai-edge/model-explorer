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
  Component,
  ViewContainerRef,
  signal,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatDialog, MatDialogModule} from '@angular/material/dialog';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import {NodeStylerDialog} from './node_styler_dialog';
import {NodeStylerService} from './node_styler_service';

/**
 * An icon that shows a node styler dialog when clicked.
 */
@Component({
  standalone: true,
  selector: 'node-styler',
  imports: [
    CommonModule,
    MatButtonModule,
    MatDialogModule,
    MatIconModule,
    MatTooltipModule,
  ],
  templateUrl: './node_styler.ng.html',
  styleUrls: ['./node_styler.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class NodeStyler {
  readonly hasNonEmptyNodeStylerRules;
  readonly dialogOpened = signal<boolean>(false);

  constructor(
    private readonly dialog: MatDialog,
    private readonly nodeStylerService: NodeStylerService,
    private readonly viewContainerRef: ViewContainerRef,
  ) {
    this.hasNonEmptyNodeStylerRules =
      this.nodeStylerService.hasNonEmptyNodeStylerRules;
  }

  handleClickOpenDialog() {
    this.dialogOpened.set(true);

    const dialogRef = this.dialog.open(NodeStylerDialog, {
      width: '800px',
      height: '600px',

      // This is needed for correctly injecting service in the
      // dialog commponent.
      viewContainerRef: this.viewContainerRef,
      hasBackdrop: false,
      autoFocus: false,
    });

    dialogRef.afterClosed().subscribe(() => {
      this.dialogOpened.set(false);
    });
  }
}
