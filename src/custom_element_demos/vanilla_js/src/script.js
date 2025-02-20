/**
 * @license
 * Copyright 2025 The Model Explorer Authors. All Rights Reserved.
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

// Set the base url of the static_files to the unpkg server.
//
// This is needed because it is not at the default location
// (the same location as the index.html file).
modelExplorer.assetFilesBaseUrl =
  'https://unpkg.com/ai-edge-model-explorer-visualizer@latest/dist/static_files';

// Prepare the graph collections.
//
// The resulting graph looks like:
// https://github.com/google-ai-edge/model-explorer/wiki/screenshots/adapter_dev_example_graph.png
//
// We are using the same example as in:
// https://github.com/google-ai-edge/model-explorer/wiki/6.-Develop-Adapter-Extension
//
// See more details in:
// https://github.com/google-ai-edge/model-explorer/blob/main/src/ui/src/components/visualizer/common/input_graph.ts
const graphCollections = [
  {
    label: 'my collection',
    graphs: [
      {
        id: 'road_trip',
        nodes: [
          // Departure city.
          {
            id: 'vancouver',
            label: 'Vancouver',
            namespace: '',
            // Node attributes.
            attrs: [
              {
                key: 'temperature',
                value: '53F',
              },
            ],
            // Outputs metadata.
            outputsMetadata: [
              {
                // Match the corresponding "sourceNodeOutputId" in Seattle's
                // incoming edge.
                id: '0',
                attrs: [
                  {
                    key: 'distance',
                    value: '230 km',
                  },
                ],
              },
              {
                // Match the corresponding "sourceNodeOutputId" in
                // SaltLakeCity's incoming edge.
                id: '1',
                attrs: [
                  {
                    key: 'distance',
                    value: '1554 km',
                  },
                ],
              },
            ],
          },

          // Coastal drive.
          // seattle -> golden gate bridge -> pier 39
          {
            id: 'seattle',
            label: 'Seattle',
            namespace: 'CoastalDrive',
            incomingEdges: [
              {
                sourceNodeId: 'vancouver',
                sourceNodeOutputId: '0',
                targetNodeInputId: '0',
              },
            ],
          },
          {
            id: 'sf_golden_gate_bridge',
            label: 'Golden gate bridge',
            namespace: 'CoastalDrive/SanFrancisco',
            incomingEdges: [
              {
                sourceNodeId: 'seattle',
                sourceNodeOutputId: '0',
                targetNodeInputId: '0',
              },
            ],
          },
          {
            id: 'sf_pier_39',
            label: 'PIER 39',
            namespace: 'CoastalDrive/SanFrancisco',
            incomingEdges: [
              {
                sourceNodeId: 'sf_golden_gate_bridge',
                sourceNodeOutputId: '0',
                targetNodeInputId: '0',
              },
            ],
          },

          // Inland drive.
          // salt lake city -> las vegas
          {
            id: 'salt_lake_city',
            label: 'Salt lake city',
            namespace: 'InlandDrive',
            incomingEdges: [
              {
                sourceNodeId: 'vancouver',
                sourceNodeOutputId: '1',
                targetNodeInputId: '0',
              },
            ],
          },
          {
            id: 'las_vegas',
            label: 'Las Vegas',
            namespace: 'InlandDrive',
            incomingEdges: [
              {
                sourceNodeId: 'salt_lake_city',
                sourceNodeOutputId: '0',
                targetNodeInputId: '0',
              },
            ],
          },

          // Destination city.
          {
            id: 'la',
            label: 'Los Angeles',
            namespace: '',
            incomingEdges: [
              {
                sourceNodeId: 'las_vegas',
                sourceNodeOutputId: '0',
                targetNodeInputId: '0',
              },
              {
                sourceNodeId: 'sf_pier_39',
                sourceNodeOutputId: '0',
                targetNodeInputId: '1',
              },
            ],
            inputsMetadata: [
              {
                // Match the first incoming edge's targetNodeInputId.
                id: '0',
                attrs: [
                  {
                    key: 'distance',
                    value: '439 km',
                  },
                ],
              },
              {
                // Match the second incoming edge's targetNodeInputId.
                id: '1',
                attrs: [
                  {
                    key: 'distance',
                    value: '613 km',
                  },
                ],
              },
            ],
          },
        ],
      },
    ],
  },
];

// Create the element with the collections above, and add to document body.
const visualizer = document.createElement('model-explorer-visualizer');
visualizer.graphCollections = graphCollections;
document.body.appendChild(visualizer);
