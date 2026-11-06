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

/** GET /api/runtime/capabilities, and GET /api/sessions/chat-options. */

/** One option a Session or a Chat may set, as the server describes it for a device: its own
 *  table of the runtime's options merged with what the last runner there reported
 *  (option_catalog.py). Values are in the record's encoding: a Session option's are text, '' for
 *  the runtime default; a Chat setting's are numbers, text, or 'on'/'off'. */
export interface OptionDescription {
  key: string;
  label: string;
  help: string;
  /** The section it is shown in. */
  group: string;
  type: 'choice' | 'integer' | 'number' | 'switch' | 'text';
  /** The values of a choice or a switch; '' is the runtime's default. */
  choices?: string[];
  /** Of a choice: what is shown for each value not shown as is, e.g. a sampler's name. */
  choice_labels?: Record<string, string>;
  /** A number's range, or a text's longest length. A Chat setting's maximum in a side's
   *  ChatCapabilities is also what that side's run bounds it to: a reply no longer than its
   *  context. */
  minimum?: number;
  maximum?: number;
  /** What a run gets when it sets none; null when the runtime decides. */
  default: string | number | null;
  /** It applies only while option `key` has `value`, e.g. CPU threads with the CPU backend. */
  requires?: {key: string; value: string};
  /** Of a sampling value: the samplers that use it. */
  samplers?: string[];
  /** Of a sampling value: the default of each sampler whose default is not `default`, such as
   *  the model's own, and where it comes from, as the server words it, e.g. "the model's
   *  default, saved with these settings". */
  defaults?: Record<string, {value: number; source: string}>;
  /** Of a sampling value: what each sampler that does not use it sends for it, as the server
   *  words it: the value the sampler fixes. */
  fixed?: Record<string, string>;
  /** Whether a run may set it; `reason` says why not. */
  available: boolean;
  reason?: string;
}
export interface RuntimeCapability {
  id: string;
  available: boolean;
  reason: string;
  backends: string[];
  /** The Session's options, in the order they are shown. */
  session_options?: OptionDescription[];
  /** The Chat's settings, in the order they are shown. */
  chat_options?: OptionDescription[];
  prompt_limit_bytes?: number;
  /** Whether a run of this runtime can capture intermediate tensors; `reason` says why not. A
   *  PyTorch run's `kind` is 'anchors': it captures every layer's four anchors by its own run
   *  options (`options`, described as a Session's options are), as `text` says in one line (how,
   *  `details`), and chooses no tensor outputs. */
  tap?: {
    available: boolean;
    reason: string;
    kind?: 'anchors';
    text?: string;
    details?: string;
    options?: OptionDescription[];
  };
  /** The Hugging Face form of the runtime (POST /api/models/huggingface). */
  huggingface?: HuggingFaceForm;
  /** Shown with the options when no runner of the device's build has run there yet: they are the
   *  server's defaults until one has. */
  notice?: string;
}
/** GET /api/sessions/chat-options?id=: what each side of a Session's chat may set, by side. */
export type ChatCapabilities = Record<string, RuntimeCapability>;
/** Where a model came from, when it came from Hugging Face (the server's hf_models). */
export interface HuggingFaceSource {
  repo: string;
  /** The .litertlm file; '' for a PyTorch model, a whole repository. */
  file: string;
  /** The revision named, 'main' unless one was. */
  revision: string;
  /** The commit it resolved to. */
  commit: string;
}
/** What the editor's Hugging Face form of a runtime says, in the server's words. */
export interface HuggingFaceForm {
  link_label: string;
  link_placeholder: string;
  /** '' for a runtime whose model is a whole repository (PyTorch). */
  file_label: string;
  file_placeholder: string;
  revision_label: string;
  revision_placeholder: string;
  help: string;
  action: string;
}
/** A model the server holds, in its workspace (GET /api/runtime/capabilities). */
export interface RuntimeModel {
  artifact: string;
  name: string;
  runtime?: string;
  /** Its bytes. */
  size?: number;
  /** Whether its file is on the server, so that a Session may load it: false for one an imported
   *  Session came with, or an older server kept outside its workspace. */
  on_server?: boolean;
  source?: HuggingFaceSource;
}
export interface RuntimeCapabilities {
  available: boolean;
  reason: string;
  models: RuntimeModel[];
  runtimes?: RuntimeCapability[];
  /** Capturing every decoder layer's anchors, what every Session does: the editor's one line of
   *  it, in the server's words. */
  tap_anchors?: {name: string; description: string};
  tap_profiles?: {id: string; name: string; description: string}[];
  platform?: string;
  capture?: string;
  upload_limit_bytes?: number;
}
