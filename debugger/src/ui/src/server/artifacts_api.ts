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

import {inject, Injectable} from '@angular/core';
import {ServerClient, type UploadOptions} from './server_client';
import type {
  ExecutionGraphCollection,
  GraphStatus,
  ModelArchitecture,
  LayerOps,
  ModelComparison,
  TapAnchors,
  TapCounterparts,
  TapScan,
} from './contracts/artifacts';
import type {RuntimeJob} from './contracts/jobs';

/** What POST /api/models/huggingface takes: a side's runtime and the model it names. */
export interface HuggingFaceRequest {
  runtime: string;
  link: string;
  file: string;
  revision: string;
}

/**
 * What POST /api/models/huggingface/check answers: whether the server has the model named, its
 * size (or what is to download) when known, and the editor's line of both.
 */
export interface HuggingFaceCheck {
  on_server: boolean;
  size: number | null;
  text: string;
}

/**
 * /api/artifacts: model files uploaded to the server, the scan of their tap points, the anchors
 * a Session of each captures by default, and their graphs: where each is, one signature's
 * graph, and the model's architecture in them.
 */
@Injectable({providedIn: 'root'})
export class ArtifactsApi {
  private readonly client = inject(ServerClient);
  /** Uploads a model file to the server and returns its artifact id. */
  upload(file: File, options: UploadOptions = {}): Promise<{artifact: string}> {
    return this.client.upload<{artifact: string}>('artifacts/upload', file, options);
  }
  /**
   * Starts the download of a model named on Hugging Face: a job, whose `model` once completed
   * is the model downloaded to the server (GET /api/jobs/{id}).
   */
  fetchHuggingFace(request: HuggingFaceRequest): Promise<RuntimeJob> {
    return this.client.post<RuntimeJob>('models/huggingface', request);
  }
  /** Whether the server has a model named on Hugging Face, and its size, before it is chosen. */
  checkHuggingFace(request: HuggingFaceRequest, signal?: AbortSignal): Promise<HuggingFaceCheck> {
    return this.client.post<HuggingFaceCheck>('models/huggingface/check', request, signal);
  }
  /** Starts scanning a model artifact's tap points. */
  startTapScan(id: string): Promise<TapScan> {
    return this.client.post<TapScan>(`artifacts/${id}/scan`, {});
  }
  /** Reads the tap scan status for a model artifact. */
  tapScan(id: string): Promise<TapScan> {
    return this.client.get<TapScan>(`artifacts/${id}/scan`);
  }
  /** Reads the default tap anchors for a model artifact. */
  tapAnchors(id: string): Promise<TapAnchors> {
    return this.client.get<TapAnchors>(`artifacts/${id}/anchors`);
  }
  /** How each point chosen of a LiteRT-LM model pairs beside the PyTorch model `pytorch`. */
  tapCounterparts(id: string, pytorch: string, points: string[]): Promise<TapCounterparts> {
    const query = new URLSearchParams({pytorch, points: points.join(',')});
    return this.client.get<TapCounterparts>(`artifacts/${id}/counterparts?${query}`);
  }
  /** What of two models compares, as the server's analyses would pair them. */
  compareModels(ref: string, target: string): Promise<ModelComparison> {
    const query = new URLSearchParams({ref, target});
    return this.client.get<ModelComparison>(`models/compare?${query}`);
  }
  /** The operators inside the layers of two models, with those ticked (`ops`). */
  layerOps(ref: string, target: string, ops: string[]): Promise<LayerOps> {
    const query = new URLSearchParams({ref, target, ops: ops.join(',')});
    return this.client.get<LayerOps>(`models/ops?${query}`);
  }
  /** Reads the execution graph export status for a model artifact. */
  graphStatus(id: string, signal?: AbortSignal): Promise<GraphStatus> {
    return this.client.get<GraphStatus>(`artifacts/${id}/graph`, signal);
  }
  /** Reads the execution graph for one signature of a model artifact. */
  signatureGraph(
    id: string,
    signature: string,
    signal?: AbortSignal,
  ): Promise<ExecutionGraphCollection> {
    return this.client.get<ExecutionGraphCollection>(
      `artifacts/${id}/graph/${encodeURIComponent(signature)}`,
      signal,
    );
  }
  /** Reads the structural model architecture for a model artifact. */
  architecture(id: string, signal?: AbortSignal): Promise<ModelArchitecture> {
    return this.client.get<ModelArchitecture>(`artifacts/${id}/architecture`, signal);
  }
}
