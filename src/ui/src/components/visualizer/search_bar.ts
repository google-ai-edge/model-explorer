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

import {ConnectedPosition, OverlaySizeConfig} from '@angular/cdk/overlay';
import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  DestroyRef,
  ElementRef,
  Input,
  ViewChild,
} from '@angular/core';
import {takeUntilDestroyed} from '@angular/core/rxjs-interop';
import {FormControl, ReactiveFormsModule} from '@angular/forms';
import {MatCheckboxModule} from '@angular/material/checkbox';
import {MatIconModule} from '@angular/material/icon';
import {MatSelectModule} from '@angular/material/select';
import {debounceTime, tap} from 'rxjs/operators';

import {Bubble} from '../bubble/bubble';

import {AppService} from './app_service';
import {type ModelGraph, ModelNode} from './common/model_graph';
import {
  SearchMatch,
  SearchMatches,
  SearchMatchType,
  SearchResults,
} from './common/types';
import {getRegexMatchesForNode, isOpNode} from './common/utils';
import {genIoTreeData, IoTree, TreeNode} from './io_tree';
import {Paginator} from './paginator';

interface SearchResultTypeOption {
  matchType: SearchMatchType;
  label: string;
  selected: boolean;
}

const TEMP_ESCAPED_SPACE = '___ESCAPED_SPACE___';

/**
 * The search bar where users search nodes/groups by keyword and show results
 * in a tree.
 */
@Component({
  standalone: true,
  selector: 'search-bar',
  imports: [
    Bubble,
    CommonModule,
    IoTree,
    MatCheckboxModule,
    MatIconModule,
    MatSelectModule,
    Paginator,
    ReactiveFormsModule,
  ],
  templateUrl: './search_bar.ng.html',
  styleUrls: ['./search_bar.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SearchBar {
  @Input({required: true}) curModelGraph!: ModelGraph;
  @Input({required: true}) rendererId!: string;
  @ViewChild('searchInput') searchInput!: ElementRef<HTMLInputElement>;
  @ViewChild('content') content!: ElementRef<HTMLElement>;

  readonly curSearchText = new FormControl<string>('');
  readonly searchResultTypes: SearchResultTypeOption[] = [
    {
      matchType: SearchMatchType.NODE_LABEL,
      label: 'Label',
      selected: true,
    },
    {
      matchType: SearchMatchType.ATTRIBUTE,
      label: 'Attrs',
      selected: true,
    },
    {
      matchType: SearchMatchType.INPUT_METADATA,
      label: 'Inputs',
      selected: true,
    },
    {
      matchType: SearchMatchType.OUTPUT_METADATA,
      label: 'Outputs',
      selected: true,
    },
  ];
  readonly searchResultTypeSelectorOverlaySize: OverlaySizeConfig = {
    minWidth: 0,
    minHeight: 0,
  };
  readonly searchResultTypeSelectorOverlayPositions: ConnectedPosition[] = [
    {
      originX: 'start',
      originY: 'bottom',
      overlayX: 'start',
      overlayY: 'top',
    },
  ];
  readonly pageSize;

  curSearchMatchedNodes: ModelNode[] = [];
  curSearchMatchData: SearchMatches[] = [];
  curSearchResultsData?: TreeNode[];
  searching = false;

  private curPageIndex = 0;

  constructor(
    private readonly appService: AppService,
    private readonly changeDetectorRef: ChangeDetectorRef,
    private readonly destroyRef: DestroyRef,
  ) {
    this.pageSize = this.appService.testMode ? 12 : 50;

    // Do search with a 300ms debounce time.
    this.curSearchText.valueChanges
      .pipe(
        tap(() => {
          this.searching = true;
          this.changeDetectorRef.markForCheck();
        }),
        debounceTime(300),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe((text) => {
        this.handleSearch((text?.toLowerCase() || '').trim());
      });

    // Focus on search input when ctrl/cmd+f is clicked.
    this.appService.searchKeyClicked
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe(() => {
        if (this.rendererId === this.appService.curSelectedRenderer()?.id) {
          this.searchInput.nativeElement.focus();
        }
      });
  }

  handleClickClearSearchText() {
    this.curSearchText.setValue('');
    this.handleSearch('');
  }

  updateSelectedResultTypes(option: SearchResultTypeOption) {
    option.selected = !option.selected;
    this.changeDetectorRef.markForCheck();
    this.handleSearch(this.curSearchText.value || '');
    setTimeout(() => {
      this.content.nativeElement.scrollTop = 0;
    });
  }

  handlePaginatorChanged(curPageIndex: number) {
    this.curPageIndex = curPageIndex;
    this.updatePagedResults();
  }

  getDisableSearchResultTypeOption(option: SearchResultTypeOption): boolean {
    return (
      this.searchResultTypes.filter((resultType) => resultType.selected)
        .length === 1 && option.selected
    );
  }

  get searchResultsTitle(): string {
    const resultCount = this.curSearchMatchedNodes?.length || 0;
    return `${resultCount} result${resultCount === 1 ? '' : 's'}`;
  }

  get showClearButton(): boolean {
    return (this.curSearchText.value || '').trim() !== '';
  }

  get showResultsPanel(): boolean {
    return (
      !this.searching &&
      (this.curSearchText.value || '').trim() !== '' &&
      !this.showNoMatches
    );
  }

  get showNoMatches(): boolean {
    return (
      !this.searching &&
      (this.curSearchText.value || '').trim() !== '' &&
      this.curSearchMatchedNodes.length === 0 &&
      this.searchResultTypes.every((type) => type.selected)
    );
  }

  get resultsCount(): number {
    return this.curSearchMatchedNodes.length;
  }

  get searchResultsContainerMaxHeight(): number {
    return document.body.offsetHeight - 300;
  }

  get showPaginator(): boolean {
    return this.resultsCount > this.pageSize;
  }

  private handleSearch(searchText: string) {
    if (!searchText) {
      this.curPageIndex = 0;
      this.curSearchResultsData = undefined;
      this.curSearchMatchedNodes = [];
      this.changeDetectorRef.markForCheck();
      this.searching = false;
      this.appService.clearSearchResults(this.rendererId);
      return;
    }

    // Filter nodes by matching its label with the search text.
    const resultNodes: ModelNode[] = [];
    const searchMatchData: SearchMatches[] = [];
    const searchResults: SearchResults = {results: {}};

    const shouldMatchTypes = new Set<SearchMatchType>(
      this.searchResultTypes
        .filter((resultType) => resultType.selected)
        .map((resultType) => resultType.matchType),
    );
    try {
      // Not using negative lookbehind to increase compatibility.
      const andParts = searchText
        .replaceAll('\\ ', TEMP_ESCAPED_SPACE)
        .split(' ')
        .filter((part) => part.trim() !== '')
        .map((part) => part.replaceAll(TEMP_ESCAPED_SPACE, '\\ '));
      const regexList = andParts.map((part) => new RegExp(part, 'i'));
      for (const node of this.curModelGraph.nodes) {
        if (isOpNode(node) && node.hideInLayout) {
          continue;
        }

        let matched = true;
        const allMatches: SearchMatch[] = [];
        const allMatchTypes = new Set<string>();
        for (const regex of regexList) {
          const {matches, matchTypes} = getRegexMatchesForNode(
            shouldMatchTypes,
            regex,
            node,
            this.curModelGraph,
            this.appService.config(),
          );
          if (matches.length === 0) {
            matched = false;
            break;
          } else {
            allMatches.push(...matches);
            for (const matchType of matchTypes) {
              allMatchTypes.add(matchType);
            }
          }
        }
        if (matched && allMatches.length > 0) {
          resultNodes.push(node);
          searchMatchData.push({
            matches: allMatches,
            matchTypes: allMatchTypes,
          });
          searchResults.results[node.id] = allMatches;
        }
      }
      this.appService.setSearchResults(this.rendererId, searchResults);
    } catch (e) {
      // Ignore.
      console.warn('Failed to search', e);
    }

    this.curPageIndex = 0;
    this.curSearchMatchedNodes = [...resultNodes];
    this.curSearchMatchData = searchMatchData;
    this.updatePagedResults();
    this.searching = false;
    this.changeDetectorRef.markForCheck();
  }

  private updatePagedResults() {
    this.curSearchResultsData = genIoTreeData(
      this.curSearchMatchedNodes.slice(
        this.curPageIndex * this.pageSize,
        (this.curPageIndex + 1) * this.pageSize,
      ),
      [],
      'incoming',
      undefined,
      this.curSearchMatchData.slice(
        this.curPageIndex * this.pageSize,
        (this.curPageIndex + 1) * this.pageSize,
      ),
    );
  }
}
