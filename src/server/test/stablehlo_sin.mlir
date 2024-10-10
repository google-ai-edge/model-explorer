#loc = loc(unknown)
#loc11 = loc("x")
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1575 : i32}, tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<f32> {mhlo.sharding = "{replicated}"} loc("x")) -> (tensor<f32> {jax.result_info = ""}) {
    %0 = stablehlo.cosine %arg0 : tensor<f32> loc(#loc19)
    %1 = stablehlo.sine %0 : tensor<f32> loc(#loc20)
    return %1 : tensor<f32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc12 = loc("<ipython-input-19-3ae4552f9a85>":8:0)
#loc19 = loc("jit(f_jax)/jit(main)/cos"(#loc12))
#loc20 = loc("jit(f_jax)/jit(main)/sin"(#loc12))