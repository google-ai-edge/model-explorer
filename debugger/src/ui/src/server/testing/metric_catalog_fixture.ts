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

import type {MetricCatalog, MetricDescription} from '../contracts/metrics';
import catalog from './metric_catalog_fixture.json';

/** The fixture's description of one metric. */
export function fixtureMetric(id: string): MetricDescription | undefined {
  return METRIC_CATALOG_FIXTURE.metrics.find((metric) => metric.id === id);
}

/** GET /api/metrics as the server answers it, for tests: TestBed specs get it through
 *  METRIC_CATALOG (src/test_providers.ts), the other specs pass it in. The JSON is the server's
 *  metric_catalog.catalog(): ci/check_server.sh checks it, and, from the repository root,
 *  `python3 -I ci/check_metric_catalog.py --write src/ui/src/server/testing/metric_catalog_fixture.json`
 *  writes it. */
export const METRIC_CATALOG_FIXTURE = catalog as MetricCatalog;
