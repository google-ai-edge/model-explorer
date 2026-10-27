// Copyright 2024 The AI Edge Model Explorer Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "utils/load_opdefs.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace model_explorer {
namespace adapter {

absl::flat_hash_map<std::string, OpMetadata> LoadTfliteOpdefs() {
  absl::flat_hash_map<std::string, OpMetadata> opdefs;
  opdefs.reserve(164);
  opdefs.emplace("abs", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("add_n", OpMetadata({"inputs"}, {"sum"}));
  opdefs.emplace("add", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("arg_max", OpMetadata({"input", "dim"}, {"output"}));
  opdefs.emplace("arg_min", OpMetadata({"input", "dim"}, {"output"}));
  opdefs.emplace("assign_variable", OpMetadata({"resource_id", "value"}, {}));
  opdefs.emplace("atan2", OpMetadata({"y", "x"}, {"output"}));
  opdefs.emplace("average_pool_2d", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("basic_lstm",
                 OpMetadata({"data_input", "prev_activ_input", "weights_input",
                             "biases_input", "prev_state_input"},
                            {"activ_output", "state_output", "concat_temp",
                             "activ_temp"}));
  opdefs.emplace("batch_matmul", OpMetadata({"x", "y"}, {"output"}));
  opdefs.emplace("batch_to_space_nd",
                 OpMetadata({"input", "block_shape", "indices"}, {"output"}));
  opdefs.emplace("bidirectional_sequence_lstm",
                 OpMetadata({"input",
                             "fw_input_to_input_weights",
                             "fw_input_to_forget_weights",
                             "fw_input_to_cell_weights",
                             "fw_input_to_output_weights",
                             "fw_recurrent_to_input_weights",
                             "fw_recurrent_to_forget_weights",
                             "fw_recurrent_to_cell_weights",
                             "fw_recurrent_to_output_weights",
                             "fw_cell_to_input_weights",
                             "fw_cell_to_forget_weights",
                             "fw_cell_to_output_weights",
                             "fw_input_gate_bias",
                             "fw_forget_gate_bias",
                             "fw_cell_bias",
                             "fw_output_gate_bias",
                             "fw_projection_weights",
                             "fw_projection_bias",
                             "bw_input_to_input_weights",
                             "bw_input_to_forget_weights",
                             "bw_input_to_cell_weights",
                             "bw_input_to_output_weights",
                             "bw_recurrent_to_input_weights",
                             "bw_recurrent_to_forget_weights",
                             "bw_recurrent_to_cell_weights",
                             "bw_recurrent_to_output_weights",
                             "bw_cell_to_input_weights",
                             "bw_cell_to_forget_weights",
                             "bw_cell_to_output_weights",
                             "bw_input_gate_bias",
                             "bw_forget_gate_bias",
                             "bw_cell_bias",
                             "bw_output_gate_bias",
                             "bw_projection_weights",
                             "bw_projection_bias",
                             "fw_input_activation_state",
                             "fw_input_cell_state",
                             "bw_input_activation_state",
                             "bw_input_cell_state",
                             "aux_input",
                             "fw_aux_input_to_input_weights",
                             "fw_aux_input_to_forget_weights",
                             "fw_aux_input_to_cell_weights",
                             "fw_aux_input_to_output_weights",
                             "bw_aux_input_to_input_weights",
                             "bw_aux_input_to_forget_weights",
                             "bw_aux_input_to_cell_weights",
                             "bw_aux_input_to_output_weights"},
                            {"fw_output", "bw_output"}));
  opdefs.emplace("bitcast", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("bitwise_xor", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("broadcast_args", OpMetadata({"s0", "s1"}, {"r0"}));
  opdefs.emplace("broadcast_to", OpMetadata({"input", "shape"}, {"output"}));
  opdefs.emplace("bucketize", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("call_once", OpMetadata({}, {}));
  opdefs.emplace("cast", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("ceil", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("complex_abs", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("concatenation", OpMetadata({"values"}, {"output"}));
  opdefs.emplace("pseudo_const", OpMetadata({}, {"output"}));
  opdefs.emplace("control_node",
                 OpMetadata({"controlInputs"}, {"outputs", "control"}));
  opdefs.emplace("conv_2d",
                 OpMetadata({"input", "filter", "bias"}, {"output"}));
  opdefs.emplace("conv_3d",
                 OpMetadata({"input", "filter", "bias"}, {"output"}));
  opdefs.emplace(
      "conv_3d_transpose",
      OpMetadata({"output_shape", "filter", "input", "bias"}, {"output"}));
  opdefs.emplace("cos", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("cumsum", OpMetadata({"input", "axis"}, {"output"}));
  opdefs.emplace("custom", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("custom_tf", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("densify", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("depth_to_space", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("depthwise_conv_2d",
                 OpMetadata({"input", "filter", "bias"}, {"output"}));
  opdefs.emplace("dequantize", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("dilate", OpMetadata({"input", "dilations", "padding_value"},
                                      {"output"}));
  opdefs.emplace("div", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace(
      "dynamic_update_slice",
      OpMetadata({"operand", "update", "start_indices"}, {"output"}));
  opdefs.emplace("elu", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("embedding_lookup",
                 OpMetadata({"lookup", "value"}, {"output"}));
  opdefs.emplace("equal", OpMetadata({"x", "y"}, {"output"}));
  opdefs.emplace("exp", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("expand_dims", OpMetadata({"input", "dim"}, {"output"}));
  opdefs.emplace("external_const", OpMetadata({}, {"output"}));
  opdefs.emplace("fake_quant", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("fill", OpMetadata({"dims", "input"}, {"result"}));
  opdefs.emplace("floor_div", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("floor_mod", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("floor", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("fully_connected",
                 OpMetadata({"input", "filter", "bias"}, {"output"}));
  opdefs.emplace("gather_nd", OpMetadata({"params", "indices"}, {"output"}));
  opdefs.emplace("gather", OpMetadata({"params", "indices"}, {"output"}));
  opdefs.emplace("gelu", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("greater_equal", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("greater", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("hard_swish", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("hashtable_find",
                 OpMetadata({"hash_table", "keys", "default_value"}, {"out"}));
  opdefs.emplace("hashtable_import",
                 OpMetadata({"hash_table", "keys", "values"}, {}));
  opdefs.emplace("hashtable", OpMetadata({}, {"out"}));
  opdefs.emplace("hashtable_size", OpMetadata({"hash_table"}, {"out"}));
  opdefs.emplace("if", OpMetadata({"cond"}, {"results"}));
  opdefs.emplace("imag", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("l2_normalization", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("lstm", OpMetadata({"input",
                                     "input_to_input_weights",
                                     "input_to_forget_weights",
                                     "input_to_cell_weights",
                                     "input_to_output_weights",
                                     "recurrent_to_input_weights",
                                     "recurrent_to_forget_weights",
                                     "recurrent_to_cell_weights",
                                     "recurrent_to_output_weights",
                                     "cell_to_input_weights",
                                     "cell_to_forget_weights",
                                     "cell_to_output_weights",
                                     "input_gate_bias",
                                     "forget_gate_bias",
                                     "cell_bias",
                                     "output_gate_bias",
                                     "projection_weights",
                                     "projection_bias",
                                     "input_activation_state",
                                     "input_cell_state",
                                     "input_layer_norm_coefficients",
                                     "forget_layer_norm_coefficients",
                                     "cell_layer_norm_coefficients",
                                     "output_layer_norm_coefficients"},
                                    {"output"}));
  opdefs.emplace("leaky_relu", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("less_equal", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("less", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("local_response_normalization",
                 OpMetadata({"input"}, {"output"}));
  opdefs.emplace("log", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("log_softmax", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("logical_and", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("logical_not", OpMetadata({"lhs"}, {"output"}));
  opdefs.emplace("logical_or", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("logistic", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("matrix_diag", OpMetadata({"diagonal"}, {"output"}));
  opdefs.emplace("matrix_set_diag",
                 OpMetadata({"input", "diagonal"}, {"result"}));
  opdefs.emplace("max_pool_2d", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("maximum", OpMetadata({"lhs", "rhs"}, {"max"}));
  opdefs.emplace("mean", OpMetadata({"input", "axis"}, {"output"}));
  opdefs.emplace("minimum", OpMetadata({"lhs", "rhs"}, {"min"}));
  opdefs.emplace("mirror_pad", OpMetadata({"input", "pad"}, {"output"}));
  opdefs.emplace("mul", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("multinomial", OpMetadata({"logits", "num_samples"}, {"out"}));
  opdefs.emplace("neg", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("no_value", OpMetadata({}, {"none_val"}));
  opdefs.emplace("non_max_suppression_v4",
                 OpMetadata({"boxes", "scores", "max_output_size",
                             "iou_threshold", "score_threshold"},
                            {"selected_indices", "valid_outputs"}));
  opdefs.emplace(
      "non_max_suppression_v5",
      OpMetadata({"boxes", "scores", "max_output_size", "iou_threshold",
                  "score_threshold", "soft_nms_sigma"},
                 {"selected_indices", "selected_scores", "valid_outputs"}));
  opdefs.emplace("not_equal", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("NumericVerify", OpMetadata({"input", "ref"}, {"output"}));
  opdefs.emplace(
      "one_hot",
      OpMetadata({"indices", "depth", "on_value", "off_value"}, {"output"}));
  opdefs.emplace("prelu", OpMetadata({"input", "alpha"}, {"output"}));
  opdefs.emplace("pack", OpMetadata({"values"}, {"output"}));
  opdefs.emplace("pad", OpMetadata({"input", "padding"}, {"output"}));
  opdefs.emplace(
      "padv2", OpMetadata({"input", "padding", "constant_values"}, {"output"}));
  opdefs.emplace("poly_call", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("pow", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("pseudo_qconst", OpMetadata({}, {"output"}));
  opdefs.emplace("quantize", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("rfft2d", OpMetadata({"input", "fft_length"}, {"output"}));
  opdefs.emplace("random_standard_normal", OpMetadata({"shape"}, {"out"}));
  opdefs.emplace("random_uniform", OpMetadata({"shape"}, {"out"}));
  opdefs.emplace("range", OpMetadata({"start", "limit", "delta"}, {"result"}));
  opdefs.emplace("rank", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("read_variable", OpMetadata({"resource_id"}, {"result"}));
  opdefs.emplace("real", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("reduce_all",
                 OpMetadata({"input", "reduction_indices"}, {"output"}));
  opdefs.emplace("reduce_any",
                 OpMetadata({"input", "reduction_indices"}, {"output"}));
  opdefs.emplace("reduce_max", OpMetadata({"input", "axes"}, {"output"}));
  opdefs.emplace("reduce_min", OpMetadata({"input", "axes"}, {"output"}));
  opdefs.emplace("reduce_prod", OpMetadata({"input", "axes"}, {"output"}));
  opdefs.emplace("relu_0_to_1", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("relu_n1_to_1", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("relu6", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("relu", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("reshape", OpMetadata({"input", "shape"}, {"output"}));
  opdefs.emplace("resize_bilinear", OpMetadata({"input", "size"}, {"output"}));
  opdefs.emplace("resize_nearest_neighbor",
                 OpMetadata({"input", "size"}, {"output"}));
  opdefs.emplace("reverse_sequence",
                 OpMetadata({"input", "seq_lengths"}, {"output"}));
  opdefs.emplace("reverse_v2", OpMetadata({"input", "axis"}, {"output"}));
  opdefs.emplace("right_shift", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("round", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("rsqrt", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("svdf", OpMetadata({"input", "feature_weights", "time_weights",
                                     "input_gate_bias", "activation_state"},
                                    {"output"}));
  opdefs.emplace("scatter_nd",
                 OpMetadata({"indices", "updates", "shape"}, {"output"}));
  opdefs.emplace("segment_sum",
                 OpMetadata({"input", "segment_ids"}, {"output"}));
  opdefs.emplace("select", OpMetadata({"condition", "x", "y"}, {"output"}));
  opdefs.emplace("select_v2", OpMetadata({"condition", "x", "y"}, {"output"}));
  opdefs.emplace("shape", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("sign", OpMetadata({"x"}, {"output"}));
  opdefs.emplace("sin", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("slice", OpMetadata({"input", "begin", "size"}, {"output"}));
  opdefs.emplace("softmax", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("space_to_batch_nd",
                 OpMetadata({"input", "block_shape", "paddings"}, {"output"}));
  opdefs.emplace("space_to_depth", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("pseudo_sparse_const", OpMetadata({}, {"output"}));
  opdefs.emplace("pseudo_sparse_qconst", OpMetadata({}, {"output"}));
  opdefs.emplace("sparse_to_dense",
                 OpMetadata({"sparse_indices", "output_shape", "sparse_values",
                             "default_value"},
                            {"dense"}));
  opdefs.emplace("split", OpMetadata({"split_dim", "value"}, {"outputs"}));
  opdefs.emplace("split_v", OpMetadata({"value", "size_splits", "split_dim"},
                                       {"outputs"}));
  opdefs.emplace("sqrt", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("square", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("squared_difference", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("squeeze", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("strided_slice",
                 OpMetadata({"input", "begin", "end", "strides"}, {"output"}));
  opdefs.emplace("sub", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("sum", OpMetadata({"input", "axes"}, {"output"}));
  opdefs.emplace("tanh", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("tile", OpMetadata({"input", "multiples"}, {"output"}));
  opdefs.emplace("topk_v2", OpMetadata({"input", "k"}, {"values", "indices"}));
  opdefs.emplace(
      "transpose_conv",
      OpMetadata({"output_shape", "weights", "input", "bias"}, {"output"}));
  opdefs.emplace("transpose", OpMetadata({"input", "perm"}, {"output"}));
  opdefs.emplace("unidirectional_sequence_lstm",
                 OpMetadata({"input",
                             "input_to_input_weights",
                             "input_to_forget_weights",
                             "input_to_cell_weights",
                             "input_to_output_weights",
                             "recurrent_to_input_weights",
                             "recurrent_to_forget_weights",
                             "recurrent_to_cell_weights",
                             "recurrent_to_output_weights",
                             "cell_to_input_weights",
                             "cell_to_forget_weights",
                             "cell_to_output_weights",
                             "input_gate_bias",
                             "forget_gate_bias",
                             "cell_bias",
                             "output_gate_bias",
                             "projection_weights",
                             "projection_bias",
                             "input_activation_state",
                             "input_cell_state",
                             "input_layer_norm_coefficients",
                             "forget_layer_norm_coefficients",
                             "cell_layer_norm_coefficients",
                             "output_layer_norm_coefficients"},
                            {"output"}));
  opdefs.emplace("unidirectional_sequence_rnn",
                 OpMetadata({"input", "input_to_input_weights",
                             "recurrent_to_input_weights", "input_gate_bias",
                             "hidden_state"},
                            {"output"}));
  opdefs.emplace("unique", OpMetadata({"input"}, {"output", "idx"}));
  opdefs.emplace("unpack", OpMetadata({"input"}, {"outputs"}));
  opdefs.emplace(
      "unsorted_segment_max",
      OpMetadata({"input", "segment_ids", "num_segments"}, {"output"}));
  opdefs.emplace(
      "unsorted_segment_min",
      OpMetadata({"input", "segment_ids", "num_segments"}, {"output"}));
  opdefs.emplace(
      "unsorted_segment_prod",
      OpMetadata({"input", "segment_ids", "num_segments"}, {"output"}));
  opdefs.emplace(
      "unsorted_segment_sum",
      OpMetadata({"input", "segment_ids", "num_segments"}, {"output"}));
  opdefs.emplace("var_handle", OpMetadata({}, {"resource_handle"}));
  opdefs.emplace("where", OpMetadata({"condition"}, {"index"}));
  opdefs.emplace("while", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("yield", OpMetadata({""}, {}));
  opdefs.emplace("zeros_like", OpMetadata({"input"}, {"output"}));
  return opdefs;
}

absl::flat_hash_map<std::string, OpMetadata> LoadStablehloOpdefs() {
  absl::flat_hash_map<std::string, OpMetadata> opdefs;
  opdefs.reserve(118);
  opdefs.emplace("stablehlo.abs", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.add", OpMetadata({"lhs", "rhs"}, {"result"}));
  opdefs.emplace("stablehlo.after_all", OpMetadata({"inputs"}, {"result"}));
  opdefs.emplace("stablehlo.all_gather", OpMetadata({"operands"}, {""}));
  opdefs.emplace("stablehlo.all_reduce", OpMetadata({"operands"}, {""}));
  opdefs.emplace("stablehlo.all_to_all", OpMetadata({"operands"}, {""}));
  opdefs.emplace("stablehlo.and", OpMetadata({"lhs", "rhs"}, {"result"}));
  opdefs.emplace("stablehlo.async_done", OpMetadata({"operand"}, {""}));
  opdefs.emplace("stablehlo.async_start", OpMetadata({"operands"}, {""}));
  opdefs.emplace("stablehlo.atan2", OpMetadata({"lhs", "rhs"}, {"result"}));
  opdefs.emplace(
      "stablehlo.batch_norm_grad",
      OpMetadata({"operand", "scale", "mean", "variance", "grad_output"},
                 {"grad_operand", "grad_scale", "grad_offset"}));
  opdefs.emplace("stablehlo.batch_norm_inference",
                 OpMetadata({"operand", "scale", "offset", "mean", "variance"},
                            {"result"}));
  opdefs.emplace("stablehlo.batch_norm_training",
                 OpMetadata({"operand", "scale", "offset"},
                            {"output", "batch_mean", "batch_var"}));
  opdefs.emplace("stablehlo.bitcast_convert", OpMetadata({"operand"}, {""}));
  opdefs.emplace("stablehlo.broadcast_in_dim", OpMetadata({"operand"}, {""}));
  opdefs.emplace("stablehlo.broadcast", OpMetadata({"operand"}, {""}));
  opdefs.emplace("stablehlo.case", OpMetadata({"index"}, {""}));
  opdefs.emplace("stablehlo.cbrt", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.ceil", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.cholesky", OpMetadata({"a"}, {"result"}));
  opdefs.emplace("stablehlo.clamp",
                 OpMetadata({"min", "operand", "max"}, {"result"}));
  opdefs.emplace("stablehlo.count_leading_zeros",
                 OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.collective_broadcast",
                 OpMetadata({"operand"}, {""}));
  opdefs.emplace("stablehlo.collective_permute", OpMetadata({"operand"}, {""}));
  opdefs.emplace("stablehlo.collective_reduce", OpMetadata({"operands"}, {""}));
  opdefs.emplace("stablehlo.compare", OpMetadata({"lhs", "rhs"}, {""}));
  opdefs.emplace("stablehlo.complex", OpMetadata({"lhs", "rhs"}, {"result"}));
  opdefs.emplace("stablehlo.composite", OpMetadata({"inputs"}, {""}));
  opdefs.emplace("stablehlo.concatenate", OpMetadata({"inputs"}, {""}));
  opdefs.emplace("stablehlo.constant", OpMetadata({}, {"output"}));
  opdefs.emplace("stablehlo.convert", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.convolution", OpMetadata({"lhs", "rhs"}, {""}));
  opdefs.emplace("stablehlo.cosine", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.create_token", OpMetadata({}, {"output"}));
  opdefs.emplace("stablehlo.cross-replica-sum", OpMetadata({"operand"}, {""}));
  opdefs.emplace("stablehlo.custom_call", OpMetadata({"inputs"}, {""}));
  opdefs.emplace("stablehlo.divide", OpMetadata({"lhs", "rhs"}, {"result"}));
  opdefs.emplace("stablehlo.dot_general", OpMetadata({"lhs", "rhs"}, {""}));
  opdefs.emplace("stablehlo.dot", OpMetadata({"lhs", "rhs"}, {""}));
  opdefs.emplace("stablehlo.dynamic_broadcast_in_dim",
                 OpMetadata({"operand", "output_dimensions"}, {""}));
  opdefs.emplace("stablehlo.dynamic_conv",
                 OpMetadata({"lhs", "rhs", "padding"}, {""}));
  opdefs.emplace("stablehlo.dynamic_gather",
                 OpMetadata({"operand", "start_indices", "slice_sizes"}, {""}));
  opdefs.emplace("stablehlo.dynamic_iota",
                 OpMetadata({"output_shape"}, {"result"}));
  opdefs.emplace("stablehlo.dynamic_pad",
                 OpMetadata({"operand", "padding_value", "edge_padding_low",
                             "edge_padding_high", "interior_padding"},
                            {"result"}));
  opdefs.emplace("stablehlo.dynamic_reshape",
                 OpMetadata({"operand", "output_shape"}, {"result"}));
  opdefs.emplace("stablehlo.dynamic_slice",
                 OpMetadata({"operand", "start_indices"}, {"result"}));
  opdefs.emplace(
      "stablehlo.dynamic_update_slice",
      OpMetadata({"operand", "update", "start_indices"}, {"result"}));
  opdefs.emplace("stablehlo.einsum", OpMetadata({"lhs", "rhs"}, {""}));
  opdefs.emplace("stablehlo.exponential", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.exponential_minus_one",
                 OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.fft", OpMetadata({"operand"}, {""}));
  opdefs.emplace("stablehlo.floor", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.gather",
                 OpMetadata({"operand", "start_indices"}, {"result"}));
  opdefs.emplace("stablehlo.get_dimension_size", OpMetadata({"operand"}, {""}));
  opdefs.emplace("stablehlo.get_tuple_element", OpMetadata({"operand"}, {""}));
  opdefs.emplace("stablehlo.if", OpMetadata({"pred"}, {""}));
  opdefs.emplace("stablehlo.imag", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.infeed", OpMetadata({"token"}, {""}));
  opdefs.emplace("stablehlo.iota", OpMetadata({}, {"output"}));
  opdefs.emplace("stablehlo.is_finite", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("stablehlo.log_plus_one", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.log", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.logistic", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.map", OpMetadata({"inputs"}, {""}));
  opdefs.emplace("stablehlo.maximum", OpMetadata({"lhs", "rhs"}, {"result"}));
  opdefs.emplace("stablehlo.minimum", OpMetadata({"lhs", "rhs"}, {"result"}));
  opdefs.emplace("stablehlo.multiply", OpMetadata({"lhs", "rhs"}, {"result"}));
  opdefs.emplace("stablehlo.negate", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.not", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.optimization_barrier",
                 OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.or", OpMetadata({"lhs", "rhs"}, {"result"}));
  opdefs.emplace("stablehlo.outfeed", OpMetadata({"inputs", "token"}, {""}));
  opdefs.emplace("stablehlo.pad",
                 OpMetadata({"operand", "padding_value"}, {""}));
  opdefs.emplace("stablehlo.partition_id", OpMetadata({}, {""}));
  opdefs.emplace("stablehlo.popcnt", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.power", OpMetadata({"lhs", "rhs"}, {"result"}));
  opdefs.emplace(
      "stablehlo.real_dynamic_slice",
      OpMetadata({"operand", "start_indices", "limit_indices", "strides"},
                 {"result"}));
  opdefs.emplace("stablehlo.real", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.recv", OpMetadata({"token"}, {""}));
  opdefs.emplace("stablehlo.reduce",
                 OpMetadata({"inputs", "init_values"}, {""}));
  opdefs.emplace("stablehlo.reduce_precision",
                 OpMetadata({"operand"}, {"output"}));
  opdefs.emplace("stablehlo.reduce_scatter", OpMetadata({"operand"}, {""}));
  opdefs.emplace("stablehlo.reduce_window",
                 OpMetadata({"inputs", "init_values"}, {""}));
  opdefs.emplace("stablehlo.remainder", OpMetadata({"lhs", "rhs"}, {"result"}));
  opdefs.emplace("stablehlo.replica_id", OpMetadata({}, {""}));
  opdefs.emplace("stablehlo.reshape", OpMetadata({"operand"}, {""}));
  opdefs.emplace("stablehlo.return", OpMetadata({"results"}, {}));
  opdefs.emplace("stablehlo.reverse", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.rng_bit_generator",
                 OpMetadata({"initial_state"}, {"output_state", "output"}));
  opdefs.emplace("stablehlo.rng", OpMetadata({"a", "b", "shape"}, {"result"}));
  opdefs.emplace("stablehlo.round_nearest_even",
                 OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.round_nearest_afz",
                 OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.rsqrt", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.scatter",
                 OpMetadata({"inputs", "scatter_indices", "updates"}, {""}));
  opdefs.emplace("stablehlo.select_and_scatter",
                 OpMetadata({"operand", "source", "init_value"}, {""}));
  opdefs.emplace("stablehlo.select",
                 OpMetadata({"pred", "on_true", "on_false"}, {"result"}));
  opdefs.emplace("stablehlo.send", OpMetadata({"inputs", "token"}, {""}));
  opdefs.emplace("stablehlo.set_dimension_size",
                 OpMetadata({"operand", "size"}, {""}));
  opdefs.emplace("stablehlo.shift_left",
                 OpMetadata({"lhs", "rhs"}, {"result"}));
  opdefs.emplace("stablehlo.shift_right_arithmetic",
                 OpMetadata({"lhs", "rhs"}, {"result"}));
  opdefs.emplace("stablehlo.shift_right_logical",
                 OpMetadata({"lhs", "rhs"}, {"result"}));
  opdefs.emplace("stablehlo.sign", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.sine", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.slice", OpMetadata({"operand"}, {""}));
  opdefs.emplace("stablehlo.sort", OpMetadata({"inputs"}, {""}));
  opdefs.emplace("stablehlo.sqrt", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.subtract", OpMetadata({"lhs", "rhs"}, {"result"}));
  opdefs.emplace("stablehlo.tan", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.tanh", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.torch_index_select",
                 OpMetadata({"operand", "index"}, {""}));
  opdefs.emplace("stablehlo.transpose", OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.triangular_solve", OpMetadata({"a", "b"}, {""}));
  opdefs.emplace("stablehlo.tuple", OpMetadata({"val"}, {"result"}));
  opdefs.emplace("stablehlo.unary_einsum", OpMetadata({"operand"}, {""}));
  opdefs.emplace("stablehlo.uniform_dequantize",
                 OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.uniform_quantize",
                 OpMetadata({"operand"}, {"result"}));
  opdefs.emplace("stablehlo.while", OpMetadata({"operand"}, {""}));
  opdefs.emplace("stablehlo.xor", OpMetadata({"lhs", "rhs"}, {"result"}));
  return opdefs;
}

}  // namespace adapter
}  // namespace model_explorer
