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
  AfterViewInit,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  ElementRef,
  HostBinding,
  Inject,
  Input,
  OnChanges,
  OnDestroy,
  ViewChild,
} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import {AppService} from './app_service';
import { ModelLoaderServiceInterface } from '../../common/model_loader_service_interface';
import type { EditableAttributeTypes, EditableValueListAttribute } from './common/input_graph';

/** Expandable info text component. */
@Component({
  selector: 'expandable-info-text',
  standalone: true,
  imports: [CommonModule, MatIconModule, MatTooltipModule],
  templateUrl: './expandable_info_text.ng.html',
  styleUrls: ['./expandable_info_text.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ExpandableInfoText implements AfterViewInit, OnDestroy, OnChanges {
  @Input() text = '';
  @Input() type = '';
  @Input() collectionLabel = '';
  @Input() nodeId = '';
  @Input() bgColor = 'transparent';
  @Input() textColor = 'inherit';
  @Input() editable?: EditableAttributeTypes = undefined;
  @ViewChild('container') container?: ElementRef<HTMLElement>;
  @ViewChild('oneLineText') oneLineText?: ElementRef<HTMLElement>;

  expanded = false;

  private hasOverflowInternal = false;
  private resizeObserver?: ResizeObserver;

  constructor(
    @Inject('ModelLoaderService')
    private readonly modelLoaderService: ModelLoaderServiceInterface,
    private readonly appService: AppService,
    private readonly changeDetectorRef: ChangeDetectorRef,
  ) {}

  @HostBinding('class.expanded') get hostExpanded() {
    return this.expanded;
  }

  ngAfterViewInit() {
    setTimeout(() => {
      this.updateHasOverflow();
      this.changeDetectorRef.markForCheck();
    });

    if (this.container) {
      this.resizeObserver = new ResizeObserver(() => {
        this.updateHasOverflow();
        this.changeDetectorRef.markForCheck();
      });
      this.resizeObserver.observe(this.container.nativeElement);
    }

    this.text = this.modelLoaderService
      .changesToUpload()[this.collectionLabel ?? '']
      ?.[this.nodeId]
      ?.find(({ key }) => key === this.type)
      ?.value ?? this.text;
  }

  ngOnChanges() {
    setTimeout(() => {
      this.updateHasOverflow();
      this.changeDetectorRef.markForCheck();
    });
  }

  ngOnDestroy() {
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }
  }

  splitEditableList(value: string) {
    return value
      .replace(/^\[/iu, '')
      .replace(/\]$/iu, '')
      .split(',')
      .map((part) => {
        const parsedValue = Number.parseFloat(part);

        if (Number.isNaN(parsedValue)) {
          return { type: 'text', value: part.trim() };
        }

        return { type: 'number', value: parsedValue }
      });
  }

  handleTextChange(evt: Event) {
    const target = evt.target as HTMLInputElement | HTMLSelectElement;
    let updatedValue = target.value;

    if (this.editable?.input_type === 'int_list') {
      updatedValue = `[${this.splitEditableList(this.text).map(({ value }, index) => {
        if (index.toString() === target.dataset['index']) {
          return target.value;
        }

        return value;
      }).join(', ')}]`;
    }

    const collectionLabel = this.appService.getSelectedPane()?.modelGraph?.collectionLabel;
    const nodeId = this.appService.getSelectedPane()?.selectedNodeInfo?.nodeId;

    this.modelLoaderService.changesToUpload.update((changesToUpload) => {
      if (collectionLabel && nodeId) {
        changesToUpload[collectionLabel] = {...changesToUpload[collectionLabel] };

        const existingChanges = changesToUpload[collectionLabel][nodeId]?.findIndex(({ key }) => key === this.type) ?? -1;

        if (existingChanges !== -1) {
          changesToUpload[collectionLabel][nodeId].splice(existingChanges, 1);
        }

        changesToUpload[collectionLabel][nodeId] = [
          ...(changesToUpload[collectionLabel][nodeId] ?? []),
          {
            key: this.type,
            value: updatedValue
          }
        ];
      }

      return changesToUpload;
    });
  }

  handleToggleExpand(event: MouseEvent, fromExpandedText = false) {
    if (!this.hasOverflow && !this.hasMultipleLines) {
      return;
    }

    event.stopPropagation();

    // Don't allow clicking on the expanded text to collapse it because users
    // might want to copy the content.
    if (fromExpandedText && this.expanded) {
      return;
    }
    this.expanded = !this.expanded;
  }

  getMaxConstValueCount(): number {
    return this.appService.config()?.maxConstValueCount ?? 0;
  }

  getEditableOptions(editable: EditableAttributeTypes, value: string) {
    return [...new Set([value, ...(editable as EditableValueListAttribute).options])];
  }

  get hasOverflow(): boolean {
    this.updateHasOverflow();
    return this.hasOverflowInternal;
  }

  get hasMultipleLines(): boolean {
    return this.type !== 'namespace' && this.text.includes('\n');
  }

  get iconName(): string {
    return this.expanded ? 'unfold_less' : 'unfold_more';
  }

  get hasBgColor(): boolean {
    return this.bgColor !== 'transparent';
  }

  get namespaceComponents(): string[] {
    const components = this.text.split('/');
    if (this.text !== '<root>') {
      components.unshift('<root>');
    }
    return components;
  }

  get formatQuantization(): string {
    const parts = this.text
      .replace('[', '')
      .replace(']', '')
      .split(',')
      .map((value) => value.trim());
    return parts.join('\n');
  }

  private updateHasOverflow() {
    if (!this.oneLineText) {
      this.hasOverflowInternal = false;
      return;
    }

    this.hasOverflowInternal =
      this.oneLineText.nativeElement.scrollWidth >
      this.oneLineText.nativeElement.offsetWidth;
    if (
      this.expanded &&
      (this.type === 'namespace' || this.type === 'values')
    ) {
      this.hasOverflowInternal = true;
    }
  }
}
