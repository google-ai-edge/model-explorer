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

import {Injectable, computed, effect, signal} from '@angular/core';
import {AppService} from './app_service';

import {ModelNode} from './common/model_graph';

import {
  NodeQuery,
  NodeQueryType,
  NodeStyleId,
  NodeStylerRule,
  NodeStylerRuleVersion,
  NodeTypeQuery,
  SearchMatchType,
  SearchNodeType,
} from './common/types';
import {
  hasNonEmptyQueries,
  matchNodeForQueries,
  processNodeStylerRules,
} from './common/utils';
import {LocalStorageService} from './local_storage_service';

/** Spec for a style. */
export interface Style {
  type: StyleType;
  label: string;
  id: NodeStyleId;
  defaultValue: string;
}

/** Style types. */
export enum StyleType {
  COLOR = 'COLOR',
  NUMBER = 'NUMBER',
}

/** Style for node background color. */
export const NODE_BG_COLOR_STYLE: Style = {
  type: StyleType.COLOR,
  label: 'Bg color',
  id: NodeStyleId.NODE_BG_COLOR,
  defaultValue: '#ffffff',
};

/** Style for node border color. */
export const NODE_BORDER_COLOR_STYLE: Style = {
  type: StyleType.COLOR,
  label: 'Border color',
  id: NodeStyleId.NODE_BORDER_COLOR,
  defaultValue: '#777777',
};

/** Style for node text color. */
export const NODE_TEXT_COLOR_STYLE: Style = {
  type: StyleType.COLOR,
  label: 'Text color',
  id: NodeStyleId.NODE_TEXT_COLOR,
  defaultValue: '#041e49',
};

/** All available styles */
export const ALL_STYLES = [
  NODE_BG_COLOR_STYLE,
  NODE_BORDER_COLOR_STYLE,
  NODE_TEXT_COLOR_STYLE,
];

const LOCAL_STORAGE_KEY_NODE_STYLER_RULES = 'model_explorer_node_styler_rules';

/**
 * Service for node styler related tasks.
 */
@Injectable()
export class NodeStylerService {
  readonly rules = signal<NodeStylerRule[]>([]);

  // Indexed by rule index then by pane index.
  readonly matchedNodes = signal<Record<number, Record<number, ModelNode[]>>>(
    {},
  );

  readonly hasNonEmptyNodeStylerRules = computed(
    () =>
      this.rules().filter(
        (rule) =>
          hasNonEmptyQueries(rule.queries) &&
          Object.keys(rule.styles).length > 0,
      ).length > 0,
  );

  constructor(
    private readonly appService: AppService,
    private readonly localStorageService: LocalStorageService,
  ) {
    effect(() => {
      const rules = this.rules();

      if (!this.appService.testMode) {
        // Save rules to local storage on changes.
        this.localStorageService.setItem(
          LOCAL_STORAGE_KEY_NODE_STYLER_RULES,
          JSON.stringify(rules),
        );
      }

      // Compute matched nodes.
      this.computeMatchedNodes(rules);
    });

    // Load rules from local storage in non-test mode.
    if (!this.appService.testMode) {
      const strRules =
        this.localStorageService.getItem(LOCAL_STORAGE_KEY_NODE_STYLER_RULES) ||
        '';
      const rules =
        strRules === '' ? [] : (JSON.parse(strRules) as NodeStylerRule[]);
      this.updateRules(rules);
    }
    // In test mode, read rules from url parameter.
    else {
      const params = new URLSearchParams(document.location.search);
      const strRules = decodeURIComponent(
        params.get('test_node_styler_rules') || '',
      );
      this.updateRules(
        strRules === '' ? [] : (JSON.parse(strRules) as NodeStylerRule[]),
      );
    }
  }

  addNewRule() {
    this.rules.update((rules) => {
      const newRules = [...rules];
      newRules.push({
        queries: [
          {
            type: NodeQueryType.NODE_TYPE,
            nodeType: SearchNodeType.OP_NODES,
          },
          {
            type: NodeQueryType.REGEX,
            queryRegex: '',
            matchTypes: [SearchMatchType.NODE_LABEL],
          },
        ],
        nodeType: SearchNodeType.OP_NODES,
        styles: {},
        version: NodeStylerRuleVersion.V2,
      });
      return newRules;
    });
  }

  moveUpRule(ruleIndex: number) {
    this.rules.update((rules) => {
      const newRules = [...rules];
      const rule = rules[ruleIndex];
      newRules.splice(ruleIndex, 1);
      newRules.splice(ruleIndex - 1, 0, rule);
      return newRules;
    });
  }

  moveDownRule(ruleIndex: number) {
    this.rules.update((rules) => {
      const newRules = [...rules];
      const rule = rules[ruleIndex];
      newRules.splice(ruleIndex, 1);
      newRules.splice(ruleIndex + 1, 0, rule);
      return newRules;
    });
  }

  duplicateRule(ruleIndex: number) {
    this.rules.update((rules) => {
      const duplicateRule = JSON.parse(
        JSON.stringify(rules[ruleIndex]),
      ) as NodeStylerRule;
      const newRules = [
        ...rules.slice(0, ruleIndex),
        duplicateRule,
        ...rules.slice(ruleIndex),
      ];
      return newRules;
    });
  }

  updateRules(rules: NodeStylerRule[]) {
    this.rules.set(this.convertOldRulesIfNecessary(rules));
  }

  updateQueries(ruleIndex: number, queries: NodeQuery[]) {
    this.rules.update((rules) => {
      const rule = rules[ruleIndex];
      rule.queries = queries;
      return [...rules];
    });
  }

  toggleStyle(ruleIndex: number, style: Style, checked?: boolean) {
    this.rules.update((rules) => {
      const rule = rules[ruleIndex];
      if (checked == null) {
        if (rule.styles[style.id] == null) {
          rule.styles[style.id] = {
            id: style.id,
            value: style.defaultValue,
          };
        } else {
          delete rule.styles[style.id];
        }
      } else {
        if (checked) {
          rule.styles[style.id] = {
            id: style.id,
            value: style.defaultValue,
          };
        } else {
          delete rule.styles[style.id];
        }
      }
      return [...rules];
    });
  }

  updateStyleValue(ruleIndex: number, style: Style, value: string) {
    this.rules.update((rules) => {
      const rule = rules[ruleIndex];
      const curStyle = rule.styles[style.id];
      if (curStyle) {
        if (typeof curStyle === 'string') {
          rule.styles[style.id] = value;
        } else {
          curStyle.value = value;
        }
      }
      return [...rules];
    });
  }

  deleteRule(ruleIndex: number) {
    this.rules.update((rules) => {
      const newRules = [...rules];
      newRules.splice(ruleIndex, 1);
      return newRules;
    });
  }

  convertOldRulesIfNecessary(rules: NodeStylerRule[]): NodeStylerRule[] {
    return rules.map((rule) => {
      // For older version of the rule, convert the node type to
      // a query.
      if (rule.version == null && rule.nodeType) {
        const nodeTypeQuery: NodeTypeQuery = {
          type: NodeQueryType.NODE_TYPE,
          nodeType: rule.nodeType,
        };
        rule.queries.unshift(nodeTypeQuery);
        rule.version = NodeStylerRuleVersion.V2;
      }
      return rule;
    });
  }

  private computeMatchedNodes(rules: NodeStylerRule[]) {
    const processedRules = processNodeStylerRules(rules);
    const matchedNodes: Record<number, Record<number, ModelNode[]>> = {};

    if (
      rules.length > 0 &&
      rules.some((rule) => hasNonEmptyQueries(rule.queries))
    ) {
      const panes = this.appService.panes();
      for (let paneIndex = 0; paneIndex < panes.length; paneIndex++) {
        const modelGraph = panes[paneIndex].modelGraph;
        if (!modelGraph) {
          continue;
        }

        for (const node of modelGraph.nodes) {
          for (
            let ruleIndex = 0;
            ruleIndex < processedRules.length;
            ruleIndex++
          ) {
            const rule = processedRules[ruleIndex];
            if (
              hasNonEmptyQueries(rules[ruleIndex].queries) &&
              matchNodeForQueries(
                node,
                rule.queries,
                modelGraph,
                this.appService.config(),
              )
            ) {
              if (matchedNodes[ruleIndex] == null) {
                matchedNodes[ruleIndex] = {};
              }
              if (matchedNodes[ruleIndex][paneIndex] == null) {
                matchedNodes[ruleIndex][paneIndex] = [];
              }
              matchedNodes[ruleIndex][paneIndex].push(node);
              break;
            }
          }
        }
      }
    }

    this.matchedNodes.set(matchedNodes);
  }
}
