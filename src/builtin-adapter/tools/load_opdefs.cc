#include "tools/load_opdefs.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace tooling {
namespace visualization_client {

absl::flat_hash_map<std::string, OpMetadata> LoadTfliteOpdefs() {
  absl::flat_hash_map<std::string, OpMetadata> opdefs;
  opdefs.reserve(164);
  opdefs.emplace("abs", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("dilate", OpMetadata({"input", "dilations", "padding_value"},
                                      {"output"}));
  opdefs.emplace("add", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("add_n", OpMetadata({"inputs"}, {"sum"}));
  opdefs.emplace("reduce_any",
                 OpMetadata({"input", "reduction_indices"}, {"output"}));
  opdefs.emplace("reduce_all",
                 OpMetadata({"input", "reduction_indices"}, {"output"}));
  opdefs.emplace(
      "transpose_conv",
      OpMetadata({"output_shape", "weights", "input", "bias"}, {"output"}));
  opdefs.emplace("average_pool_2d", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("arg_max", OpMetadata({"input", "dim"}, {"output"}));
  opdefs.emplace("arg_min", OpMetadata({"input", "dim"}, {"output"}));
  opdefs.emplace("ceil", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("concatenation", OpMetadata({"values"}, {"output"}));
  opdefs.emplace("pseudo_const", OpMetadata({}, {"output"}));
  opdefs.emplace("pseudo_sparse_const", OpMetadata({}, {"output"}));
  opdefs.emplace("external_const", OpMetadata({}, {"output"}));
  opdefs.emplace("conv_2d", OpMetadata({"input", "filter", "bias"}, {}));
  opdefs.emplace("cos", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("cumsum", OpMetadata({"input", "axis"}, {"output"}));
  opdefs.emplace("depthwise_conv_2d",
                 OpMetadata({"input", "filter", "bias"}, {}));
  opdefs.emplace("fully_connected",
                 OpMetadata({"input", "filter", "bias"}, {"output"}));
  opdefs.emplace("batch_matmul", OpMetadata({"x", "y"}, {"output"}));
  opdefs.emplace("gather", OpMetadata({"params", "indices"}, {"output"}));
  opdefs.emplace("gather_nd", OpMetadata({"params", "indices"}, {"output"}));
  opdefs.emplace("scatter_nd",
                 OpMetadata({"indices", "updates", "shape"}, {"output"}));
  opdefs.emplace("less_equal", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("local_response_normalization",
                 OpMetadata({"input"}, {"output"}));
  opdefs.emplace("greater_equal", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("matrix_diag", OpMetadata({"diagonal"}, {"output"}));
  opdefs.emplace("matrix_set_diag",
                 OpMetadata({"input", "diagonal"}, {"result"}));
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
  opdefs.emplace("div", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("elu", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("embedding_lookup",
                 OpMetadata({"lookup", "value"}, {"output"}));
  opdefs.emplace("equal", OpMetadata({"x", "y"}, {"output"}));
  opdefs.emplace("exp", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("expand_dims", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("squeeze", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("fill", OpMetadata({"dims", "input"}, {"result"}));
  opdefs.emplace("floor", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("floor_div", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("floor_mod", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("greater", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("hard_swish", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("l2_normalization", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("leaky_relu", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("less", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("logical_and", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("logical_not", OpMetadata({"lhs"}, {"output"}));
  opdefs.emplace("logical_or", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("logistic", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("log", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("log_softmax", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("max_pool_2d", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("maximum", OpMetadata({"lhs", "rhs"}, {"max"}));
  opdefs.emplace("mean", OpMetadata({"input", "axis"}, {"output"}));
  opdefs.emplace(
      "one_hot",
      OpMetadata({"indices", "depth", "on_value", "off_value"}, {"output"}));
  opdefs.emplace("round", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("slice", OpMetadata({"input", "begin", "size"}, {"output"}));
  opdefs.emplace("sum", OpMetadata({"input", "axes"}, {"output"}));
  opdefs.emplace("reduce_min", OpMetadata({"input", "axes"}, {"output"}));
  opdefs.emplace("reduce_max", OpMetadata({"input", "axes"}, {"output"}));
  opdefs.emplace("reduce_prod", OpMetadata({"input", "axes"}, {"output"}));
  opdefs.emplace("minimum", OpMetadata({"lhs", "rhs"}, {"min"}));
  opdefs.emplace("mul", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("neg", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("pack", OpMetadata({"values"}, {"output"}));
  opdefs.emplace("pad", OpMetadata({"input", "padding"}, {"output"}));
  opdefs.emplace(
      "padv2", OpMetadata({"input", "padding", "constant_values"}, {"output"}));
  opdefs.emplace("poly_call", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("pow", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("prelu", OpMetadata({"input", "alpha"}, {"output"}));
  opdefs.emplace("rank", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("relu", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("relu6", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("relu_0_to_1", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("relu_n1_to_1", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("reshape", OpMetadata({"input", "shape"}, {"output"}));
  opdefs.emplace("reverse_sequence",
                 OpMetadata({"input", "seq_lengths"}, {"output"}));
  opdefs.emplace("rsqrt", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("shape", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("range", OpMetadata({"start", "limit", "delta"}, {"result"}));
  opdefs.emplace("reverse_v2", OpMetadata({"input", "axis"}, {"output"}));
  opdefs.emplace("select", OpMetadata({"condition", "x", "y"}, {"output"}));
  opdefs.emplace("select_v2", OpMetadata({"condition", "x", "y"}, {"output"}));
  opdefs.emplace("sin", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("softmax", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("sqrt", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("square", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("sub", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("squared_difference", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("tanh", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("tile", OpMetadata({"input", "multiples"}, {"output"}));
  opdefs.emplace("topk_v2", OpMetadata({"input", "k"}, {"values", "indices"}));
  opdefs.emplace("transpose", OpMetadata({"input", "perm"}, {"output"}));
  opdefs.emplace("unpack", OpMetadata({"input"}, {"outputs"}));
  opdefs.emplace("zeros_like", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("batch_to_space_nd",
                 OpMetadata({"input", "block_shape", "indices"}, {"output"}));
  opdefs.emplace("space_to_batch_nd",
                 OpMetadata({"input", "block_shape", "paddings"}, {"output"}));
  opdefs.emplace("space_to_depth", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("depth_to_space", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("split", OpMetadata({"split_dim", "value"}, {"outputs"}));
  opdefs.emplace("split_v", OpMetadata({"value", "size_splits", "split_dim"},
                                       {"outputs"}));
  opdefs.emplace("resize_bilinear", OpMetadata({"input", "size"}, {"output"}));
  opdefs.emplace("resize_nearest_neighbor",
                 OpMetadata({"input", "size"}, {"output"}));
  opdefs.emplace("sparse_to_dense",
                 OpMetadata({"sparse_indices", "output_shape", "sparse_values",
                             "default_value"},
                            {"dense"}));
  opdefs.emplace("strided_slice",
                 OpMetadata({"input", "begin", "end", "strides"}, {"output"}));
  opdefs.emplace("cast", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("mirror_pad", OpMetadata({"input", "pad"}, {"output"}));
  opdefs.emplace("unique", OpMetadata({"input"}, {"output", "idx"}));
  opdefs.emplace("gelu", OpMetadata({"input"}, {"output"}));
  opdefs.emplace(
      "dynamic_update_slice",
      OpMetadata({"operand", "update", "start_indices"}, {"output"}));
  opdefs.emplace("bitcast", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("bitwise_xor", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("right_shift", OpMetadata({"lhs", "rhs"}, {"output"}));
  opdefs.emplace("dequantize", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("fake_quant", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("pseudo_qconst", OpMetadata({"qtype"}, {"output"}));
  opdefs.emplace("pseudo_sparse_qconst", OpMetadata({"qtype"}, {"output"}));
  opdefs.emplace("quantize", OpMetadata({"input", "qtype"}, {"output"}));
  opdefs.emplace("densify", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("basic_lstm",
                 OpMetadata({"data_input", "prev_activ_input", "weights_input",
                             "biases_input", "prev_state_input"},
                            {"activ_output", "state_output", "concat_temp",
                             "activ_temp"}));
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
  opdefs.emplace("unidirectional_sequence_rnn",
                 OpMetadata({"input", "input_to_input_weights",
                             "recurrent_to_input_weights", "input_gate_bias",
                             "hidden_state"},
                            {"output"}));
  opdefs.emplace("where", OpMetadata({"condition"}, {"index"}));
  opdefs.emplace("NumericVerify", OpMetadata({"input", "ref"}, {"output"}));
  opdefs.emplace("svdf", OpMetadata({"input", "feature_weights", "time_weights",
                                     "input_gate_bias", "activation_state"},
                                    {"output"}));
  opdefs.emplace("segment_sum",
                 OpMetadata({"input", "segment_ids"}, {"output"}));
  opdefs.emplace(
      "unsorted_segment_prod",
      OpMetadata({"input", "segment_ids", "num_segments"}, {"output"}));
  opdefs.emplace(
      "unsorted_segment_max",
      OpMetadata({"input", "segment_ids", "num_segments"}, {"output"}));
  opdefs.emplace(
      "unsorted_segment_min",
      OpMetadata({"input", "segment_ids", "num_segments"}, {"output"}));
  opdefs.emplace(
      "unsorted_segment_sum",
      OpMetadata({"input", "segment_ids", "num_segments"}, {"output"}));
  opdefs.emplace("atan2", OpMetadata({"y", "x"}, {"output"}));
  opdefs.emplace("sign", OpMetadata({"x"}, {"output"}));
  opdefs.emplace("yield", OpMetadata({}, {}));
  opdefs.emplace("if", OpMetadata({"cond"}, {"results"}));
  opdefs.emplace("while", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("call_once", OpMetadata({}, {}));
  opdefs.emplace("custom", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("custom_tf", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("broadcast_to", OpMetadata({"input", "shape"}, {"output"}));
  opdefs.emplace("rfft2d", OpMetadata({"input", "fft_length"}, {"output"}));
  opdefs.emplace("var_handle", OpMetadata({}, {"resource_handle"}));
  opdefs.emplace("assign_variable", OpMetadata({"resource_id", "value"}, {}));
  opdefs.emplace("read_variable", OpMetadata({"resource_id"}, {"result"}));
  opdefs.emplace("conv_3d",
                 OpMetadata({"input", "filter", "bias"}, {"output"}));
  opdefs.emplace(
      "conv_3d_transpose",
      OpMetadata({"output_shape", "filter", "input", "bias"}, {"output"}));
  opdefs.emplace("complex_abs", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("real", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("imag", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("hashtable", OpMetadata({}, {"out"}));
  opdefs.emplace("hashtable_find",
                 OpMetadata({"hash_table", "keys", "default_value"}, {"out"}));
  opdefs.emplace("hashtable_import",
                 OpMetadata({"hash_table", "keys", "values"}, {}));
  opdefs.emplace("hashtable_size", OpMetadata({"hash_table"}, {"out"}));
  opdefs.emplace("broadcast_args", OpMetadata({"s0", "s1"}, {"r0"}));
  opdefs.emplace("bucketize", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("random_uniform", OpMetadata({"shape"}, {"out"}));
  opdefs.emplace("random_standard_normal", OpMetadata({"shape"}, {"out"}));
  opdefs.emplace("multinomial", OpMetadata({"logits", "num_samples"}, {"out"}));
  opdefs.emplace("no_value", OpMetadata({}, {}));
  opdefs.emplace("control_node", OpMetadata({}, {"outputs"}));
  return opdefs;
}

absl::flat_hash_map<std::string, OpMetadata> LoadTfOpdefs() {
  absl::flat_hash_map<std::string, OpMetadata> opdefs;
  opdefs.reserve(89);
  opdefs.emplace("Case", OpMetadata({"branch_index", "input"}, {"output"}));
  opdefs.emplace("CaseRegion", OpMetadata({"branch_index"}, {"output"}));
  opdefs.emplace("Const", OpMetadata({}, {"output"}));
  opdefs.emplace("EmptyTensorList",
                 OpMetadata({"element_shape", "max_num_elements"}, {}));
  opdefs.emplace("If", OpMetadata({"cond", "input"}, {"output"}));
  opdefs.emplace("Yield", OpMetadata({}, {}));
  opdefs.emplace("IfRegion", OpMetadata({"cond"}, {"output"}));
  opdefs.emplace("GeneratorDatasetRegion",
                 OpMetadata({"init_func_other_args", "next_func_other_args",
                             "finalize_func_other_args"},
                            {"handle"}));
  opdefs.emplace("LegacyCall", OpMetadata({"args"}, {"output"}));
  opdefs.emplace("ParseExample",
                 OpMetadata({"serialized", "names", "sparse_keys", "dense_keys",
                             "dense_defaults"},
                            {}));
  opdefs.emplace("ParseExampleV2",
                 OpMetadata({"serialized", "names", "sparse_keys", "dense_keys",
                             "ragged_keys", "dense_defaults"},
                            {}));
  opdefs.emplace("Placeholder", OpMetadata({}, {"output"}));
  opdefs.emplace("PlaceholderWithDefault", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("StatefulPartitionedCall", OpMetadata({"args"}, {"output"}));
  opdefs.emplace("While", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("WhileRegion", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("TensorListReserve",
                 OpMetadata({"element_shape", "num_elements"}, {}));
  opdefs.emplace("VarHandleOp", OpMetadata({}, {"resource"}));
  opdefs.emplace("XlaSharding", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("InfeedDequeueTuple", OpMetadata({}, {"outputs"}));
  opdefs.emplace("InfeedEnqueueTuple", OpMetadata({"inputs"}, {}));
  opdefs.emplace("StringFormat", OpMetadata({"inputs"}, {"output"}));
  opdefs.emplace("ReduceDataset", OpMetadata({"input_dataset", "initial_state",
                                              "other_arguments"},
                                             {"components"}));
  opdefs.emplace("ToBool", OpMetadata({"input"}, {"output"}));
  opdefs.emplace("BesselI0e", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("BesselI1e", OpMetadata({"x"}, {"y"}));
  opdefs.emplace("TPUPartitionedCall",
                 OpMetadata({"args", "device_ordinal"}, {"output"}));
  opdefs.emplace("StatefulUniformFullInt",
                 OpMetadata({"resource", "algorithm", "shape"}, {"output"}));
  opdefs.emplace("StatefulUniformInt", OpMetadata({"resource", "algorithm",
                                                   "shape", "minval", "maxval"},
                                                  {"output"}));
  opdefs.emplace("CloseSummaryWriter", OpMetadata({"writer"}, {}));
  opdefs.emplace("CreateSummaryDbWriter",
                 OpMetadata({"writer", "db_uri", "experiment_name", "run_name",
                             "user_name"},
                            {}));
  opdefs.emplace("CreateSummaryFileWriter",
                 OpMetadata({"writer", "logdir", "max_queue", "flush_millis",
                             "filename_suffix"},
                            {}));
  opdefs.emplace("FlushSummaryWriter", OpMetadata({"writer"}, {}));
  opdefs.emplace("ImportEvent", OpMetadata({"writer", "event"}, {}));
  opdefs.emplace("SummaryWriter", OpMetadata({}, {"writer"}));
  opdefs.emplace(
      "WriteAudioSummary",
      OpMetadata({"writer", "step", "tag", "tensor", "sample_rate"}, {}));
  opdefs.emplace("WriteGraphSummary",
                 OpMetadata({"writer", "step", "tensor"}, {}));
  opdefs.emplace("WriteHistogramSummary",
                 OpMetadata({"writer", "step", "tag", "values"}, {}));
  opdefs.emplace(
      "WriteImageSummary",
      OpMetadata({"writer", "step", "tag", "tensor", "bad_color"}, {}));
  opdefs.emplace("WriteRawProtoSummary",
                 OpMetadata({"writer", "step", "tensor"}, {}));
  opdefs.emplace("WriteScalarSummary",
                 OpMetadata({"writer", "step", "tag", "value"}, {}));
  opdefs.emplace(
      "WriteSummary",
      OpMetadata({"writer", "step", "tensor", "tag", "summary_metadata"}, {}));
  opdefs.emplace("_TPUDeviceOrdinalPlaceholder",
                 OpMetadata({}, {"device_ordinal"}));
  opdefs.emplace("TPUPartitionedInput", OpMetadata({"inputs"}, {"output"}));
  opdefs.emplace("TPUPartitionedInputV2", OpMetadata({"inputs"}, {"output"}));
  opdefs.emplace("TPUPartitionedOutput", OpMetadata({"inputs"}, {"output"}));
  opdefs.emplace("TPUPartitionedOutputV2", OpMetadata({"inputs"}, {"output"}));
  opdefs.emplace("XlaHostCompute", OpMetadata({"inputs"}, {"outputs"}));
  opdefs.emplace("ConfigureAndInitializeGlobalTPU", OpMetadata({}, {"output"}));
  opdefs.emplace("ShutdownTPUSystem", OpMetadata({}, {"success"}));
  opdefs.emplace("_InternalTestNonResourceValueSideEffects_",
                 OpMetadata({"key"}, {}));
  opdefs.emplace("_InternalTestMustExecuteTrait_", OpMetadata({}, {}));
  opdefs.emplace("SetStaticDimensionBounds",
                 OpMetadata({"input", "static_shape"}, {"output"}));
  opdefs.emplace("TPUCompileMlirAndExecute",
                 OpMetadata({"args", "static_shapes"},
                            {"rendezvous_key_base", "results"}));
  opdefs.emplace("CollateTPUEmbeddingMemory",
                 OpMetadata({"memory_configs"}, {"merged_memory_config"}));
  opdefs.emplace("BatchNormWithGlobalNormalization",
                 OpMetadata({"x", "m", "v", "beta", "gamma"}, {"result"}));
  opdefs.emplace(
      "ClipByValue",
      OpMetadata({"x", "clip_value_min", "clip_value_max"}, {"output"}));
  opdefs.emplace("L2Loss", OpMetadata({"x"}, {"output"}));
  opdefs.emplace("Select", OpMetadata({"condition", "then_value", "else_value"},
                                      {"output"}));
  opdefs.emplace(
      "SelectV2",
      OpMetadata({"condition", "then_value", "else_value"}, {"output"}));
  opdefs.emplace("TPUCopyWithDynamicShape",
                 OpMetadata({"tensors", "unpadded_sizes"}, {"tpu_tensors"}));
  opdefs.emplace("TPUAnnotateTensorsWithDynamicShape",
                 OpMetadata({"tensors"}, {"tpu_tensors"}));
  opdefs.emplace("ConvertToCooTensor",
                 OpMetadata({"indices_or_row_splits", "values", "weights"},
                            {"row_ids", "col_ids", "gains"}));
  opdefs.emplace("GlobalIterId", OpMetadata({}, {"iter_id"}));
  opdefs.emplace("ResourceGatherNd",
                 OpMetadata({"resource", "indices"}, {"output"}));
  opdefs.emplace(
      "XlaSparseCoreAdagrad",
      OpMetadata({"indices", "gradient", "learning_rate", "accumulator",
                  "embedding_table"},
                 {"updated_embedding_table", "updated_accumulator"}));
  opdefs.emplace(
      "XlaSparseCoreAdagradMomentum",
      OpMetadata({"indices", "gradient", "learning_rate", "beta_1", "epsilon",
                  "accumulator", "momentum", "embedding_table"},
                 {"updated_embedding_table", "updated_accumulator",
                  "updated_momentum"}));
  opdefs.emplace(
      "XlaSparseCoreAdam",
      OpMetadata(
          {"embedding_table", "indices", "gradient", "learning_rate",
           "momentum", "velocity", "beta_1", "beta_2", "epsilon"},
          {"updated_embedding_table", "updated_velocity", "updated_momentum"}));
  opdefs.emplace(
      "XlaSparseCoreFtrl",
      OpMetadata({"embedding_table", "accumulator", "linear", "learning_rate",
                  "indices", "gradient", "beta", "learning_rate_power",
                  "l2_regularization_strength"},
                 {"updated_embedding_table", "updated_accumulator",
                  "updated_linear"}));
  opdefs.emplace(
      "XlaSparseCoreSgd",
      OpMetadata({"indices", "gradient", "learning_rate", "embedding_table"},
                 {"updated_embedding_table"}));
  opdefs.emplace(
      "XlaSparseDenseMatmulWithCsrInput",
      OpMetadata({"row_pointers", "sorted_sample_ids", "sorted_token_ids",
                  "sorted_gains", "embedding_table",
                  "num_minibatches_per_physical_sparse_core"},
                 {"activations"}));
  opdefs.emplace(
      "XlaSparseDenseMatmulGradWithSgdAndCsrInput",
      OpMetadata(
          {"row_pointers", "sorted_sample_ids", "sorted_token_ids",
           "sorted_gains", "activation_gradients", "learning_rate",
           "embedding_table", "num_minibatches_per_physical_sparse_core"},
          {"updated_embedding_table"}));
  opdefs.emplace(
      "XlaSparseDenseMatmulGradWithAdagradAndCsrInput",
      OpMetadata({"row_pointers", "sorted_sample_ids", "sorted_token_ids",
                  "sorted_gains", "activation_gradients", "learning_rate",
                  "embedding_table", "accumulator",
                  "num_minibatches_per_physical_sparse_core"},
                 {"updated_embedding_table", "updated_accumulator"}));
  opdefs.emplace(
      "XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInput",
      OpMetadata({"row_pointers", "sorted_sample_ids", "sorted_token_ids",
                  "sorted_gains", "activation_gradients", "learning_rate",
                  "embedding_table", "accumulator", "momenta",
                  "num_minibatches_per_physical_sparse_core"},
                 {"updated_embedding_table", "updated_accumulator",
                  "updated_momenta"}));
  opdefs.emplace(
      "XlaSparseDenseMatmulGradWithAdamAndCsrInput",
      OpMetadata(
          {"row_pointers", "sorted_sample_ids", "sorted_token_ids",
           "sorted_gains", "activation_gradients", "learning_rate",
           "embedding_table", "momenta", "velocity",
           "num_minibatches_per_physical_sparse_core"},
          {"updated_embedding_table", "updated_momenta", "updated_velocity"}));
  opdefs.emplace(
      "XlaSparseDenseMatmulGradWithFtrlAndCsrInput",
      OpMetadata({"row_pointers", "sorted_sample_ids", "sorted_token_ids",
                  "sorted_gains", "activation_gradients", "learning_rate",
                  "embedding_table", "accumulator", "linear",
                  "num_minibatches_per_physical_sparse_core"},
                 {"updated_embedding_table", "updated_accumulator",
                  "updated_linear"}));
  opdefs.emplace(
      "GetMinibatchesInCsrWithPhysicalReplica",
      OpMetadata(
          {"program_key", "row_ids", "col_ids", "gains", "splits", "id_counts"},
          {"row_pointers", "sorted_sample_ids", "sorted_token_ids",
           "sorted_gains", "row_pointers_unpadded_size", "ids_unpadded_size",
           "num_minibatches_per_physical_sparse_core"}));
  opdefs.emplace("GetMinibatchSplitsWithPhysicalReplica",
                 OpMetadata({"program_key", "row_ids", "col_ids", "gains"},
                            {"sorted_row_ids", "sorted_col_ids", "sorted_gains",
                             "splits", "id_counts", "max_ids", "max_uniques"}));
  opdefs.emplace("StoreMinibatchStatisticsInFdo",
                 OpMetadata({"program_key", "max_ids", "max_uniques"}, {}));
  opdefs.emplace("ConvertToListOfSparseCoreCooTensors",
                 OpMetadata({"indices_or_row_splits", "values", "weights"},
                            {"row_ids_list", "col_ids_list", "gains_list"}));
  opdefs.emplace("SortListOfSparseCoreCooTensors",
                 OpMetadata({"row_ids_list", "col_ids_list", "gains_list"},
                            {"sorted_row_ids", "sorted_col_ids", "sorted_gains",
                             "id_counts"}));
  opdefs.emplace(
      "ConvertToSparseCoreCsrWrappedCooTensorOp",
      OpMetadata({"sorted_row_ids_list", "sorted_col_ids_list",
                  "sorted_gains_list", "id_counts_list", "splits"},
                 {"row_pointers", "sorted_sample_ids", "sorted_token_ids",
                  "sorted_gains", "row_pointers_unpadded_size",
                  "ids_unpadded_size", "num_minibatches_per_sc"}));
  opdefs.emplace("GetStatsFromListOfSparseCoreCooTensors",
                 OpMetadata({"row_ids_list", "col_ids_list", "gains_list"},
                            {"max_ids_per_sparse_core",
                             "max_unique_ids_per_sparse_core"}));
  opdefs.emplace(
      "XlaSparseDenseMatmulWithStaticBufferSize",
      OpMetadata({"row_pointers", "sorted_sample_ids", "sorted_token_ids",
                  "sorted_gains", "embedding_table",
                  "num_minibatches_per_physical_sparse_core"},
                 {"activations"}));
  opdefs.emplace(
      "XlaSparseDenseMatmulGradWithSgdAndStaticBufferSize",
      OpMetadata(
          {"row_pointers", "sorted_sample_ids", "sorted_token_ids",
           "sorted_gains", "activation_gradients", "learning_rate",
           "embedding_table", "num_minibatches_per_physical_sparse_core"},
          {"updated_embedding_table"}));
  opdefs.emplace(
      "XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSize",
      OpMetadata({"row_pointers", "sorted_sample_ids", "sorted_token_ids",
                  "sorted_gains", "activation_gradients", "learning_rate",
                  "embedding_table", "accumulator",
                  "num_minibatches_per_physical_sparse_core"},
                 {"updated_embedding_table", "updated_accumulator"}));
  opdefs.emplace(
      "XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSize",
      OpMetadata({"row_pointers", "sorted_sample_ids", "sorted_token_ids",
                  "sorted_gains", "activation_gradients", "learning_rate",
                  "embedding_table", "accumulator", "momenta",
                  "num_minibatches_per_physical_sparse_core"},
                 {"updated_embedding_table", "updated_accumulator",
                  "updated_momenta"}));
  opdefs.emplace(
      "XlaSparseDenseMatmulGradWithAdamAndStaticBufferSize",
      OpMetadata(
          {"row_pointers", "sorted_sample_ids", "sorted_token_ids",
           "sorted_gains", "activation_gradients", "learning_rate",
           "embedding_table", "momenta", "velocity",
           "num_minibatches_per_physical_sparse_core"},
          {"updated_embedding_table", "updated_momenta", "updated_velocity"}));
  opdefs.emplace(
      "XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSize",
      OpMetadata({"row_pointers", "sorted_sample_ids", "sorted_token_ids",
                  "sorted_gains", "activation_gradients", "learning_rate",
                  "embedding_table", "accumulator", "linear",
                  "num_minibatches_per_physical_sparse_core"},
                 {"updated_embedding_table", "updated_accumulator",
                  "updated_linear"}));
  return opdefs;
}

}  // namespace visualization_client
}  // namespace tooling
