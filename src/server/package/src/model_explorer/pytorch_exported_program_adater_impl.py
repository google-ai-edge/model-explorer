import json
import types
from typing import Dict

import torch
import torch.fx
from torch.fx import _pytree as fx_pytree

from .graph_builder import Graph, GraphNode, IncomingEdge, KeyValue, MetadataItem
from .types import ModelExplorerGraphs


class PytorchExportedProgramAdapterImpl:

  def __init__(self, ep: torch.export.ExportedProgram):
    self.ep = ep
    self.gm = self.ep.graph_module
    self.inputs_map = self.get_inputs_map()

  def _graph_module_flat_inputs(self, ep: torch.export.ExportedProgram, args, kwargs):
    """Transform args, kwargs of __call__ to args for graph_module.

    self.graph_module takes stuff from state dict as inputs.
    The invariant is for ep: ExportedProgram is
    ep(args, kwargs) ==
      ep.postprocess(ep.graph_module(ep.graph_module_flat_inputs(args, kwargs)))
    """
    if args is None:
      args = tuple()
    if kwargs is None:
      kwargs = {}

    flat_args = args
    if (in_spec := ep.call_spec.in_spec) is not None:
      if (
          in_spec.type == tuple
          and len(in_spec.children_specs) == 2
          and in_spec.children_specs[0].type == tuple
          and in_spec.children_specs[1].type == dict
      ):
        # NOTE: this is the case where in_spec is for both args and kwargs
        flat_args = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
      else:
        flat_args = fx_pytree.tree_flatten_spec(args, in_spec)

    param_buffer_keys = (
        ep.graph_signature.parameters + ep.graph_signature.buffers
    )
    param_buffer_values = tuple(
        ep.state_dict[key] for key in param_buffer_keys)

    if hasattr(ep.graph_signature, 'lifted_tensor_constants'):
      ordered_tensor_constants = tuple(
          ep.tensor_constants[name]
          for name in ep.graph_signature.lifted_tensor_constants
      )
    else:
      ordered_tensor_constants = tuple()

    return (*param_buffer_values, *flat_args, *ordered_tensor_constants)

  def get_inputs_map(self):
    inputs_map = {}
    if not self.ep.example_inputs:
      print(
          'WARNING: no ExportedProgram.example_inputs found. Cannot show'
          ' constant tensor values in Model Explorer.'
      )
      return inputs_map

    input_tensors = self._graph_module_flat_inputs(
        self.ep, *self.ep.example_inputs
    )
    for input_spec, tensor in zip(
        self.ep.graph_signature.input_specs, input_tensors
    ):
      inputs_map[input_spec.arg.name] = [input_spec.target, tensor]
    return inputs_map

  def is_arg_node(self, fx_node: torch.fx.node.Node):
    return fx_node.op == 'placeholder'

  def is_getitem_node(self, fx_node: torch.fx.node.Node):
    return isinstance(fx_node.target, types.BuiltinFunctionType)

  def class_fullname(self, klass):
    module = klass.__module__
    if module == 'builtins':
      return klass.__qualname__
    return module + '.' + klass.__qualname__

  def get_label(self, fx_node: torch.fx.node.Node):
    if hasattr(fx_node.target, 'overloadpacket'):
      return str(fx_node.target.overloadpacket)
    if self.is_getitem_node(fx_node):
      return 'getitem'
    return str(fx_node.target)

  def get_hierachy(self, fx_node: torch.fx.node.Node):
    # Stores all arg and input nodes to `inputs` namespace.
    if self.is_arg_node(fx_node):
      return 'inputs'

    stack_traces = fx_node.meta.get('nn_module_stack', {})
    layers = []
    for name, layer in stack_traces.values():
      iid = '' if not name else '_' + name.split('.')[-1]
      layer_str = (
          layer if isinstance(layer, str) else self.class_fullname(layer)
      )
      layers.append(layer_str + iid)
    hierachy_str = '/'.join(layers)
    return hierachy_str

  def add_incoming_edges(self, fx_node: torch.fx.node.Node, node: GraphNode):
    for target_input_id, input_fx_node in enumerate(fx_node.all_input_nodes):
      source_node_output_id = '0'  # default to the first output
      for idx, user in enumerate(input_fx_node.users):
        if user == fx_node:
          source_node_output_id = str(idx)
          break
      node.incomingEdges.append(
          IncomingEdge(
              sourceNodeId=input_fx_node.name,
              sourceNodeOutputId=source_node_output_id,
              targetNodeInputId=str(target_input_id),
          )
      )

  def print_tensor(self, tensor: torch.Tensor, size_limit: int = 16):
    shape = tensor.shape
    total_size = 1
    for dim in shape:
      total_size *= dim
    if size_limit < 0 or size_limit > total_size:
      return json.dumps(tensor.detach().numpy().tolist())

    return json.dumps((tensor.detach().numpy().flatten())[:size_limit].tolist())

  def add_node_attrs(self, fx_node: torch.fx.node.Node, node: GraphNode):
    if hasattr(fx_node.target, '_schema'):
      for idx, arg in enumerate(fx_node.target._schema.arguments):
        if idx < len(fx_node.args):
          node.attrs.append(
              KeyValue(key=arg.name, value=str(fx_node.args[idx]))
          )
        else:
          val = fx_node.kwargs.get(arg.name, arg.default_value)
          node.attrs.append({'key': arg.name, 'value': str(val)})

    if self.is_arg_node(fx_node):
      tensor_spec = self.inputs_map.get(fx_node.name)
      if tensor_spec:
        node.attrs.append(KeyValue(key='target', value=str(tensor_spec[0])))
        node.attrs.append(
            KeyValue(key='__value', value=self.print_tensor(tensor_spec[1]))
        )

  def add_outputs_metadata(self, fx_node: torch.fx.node.Node, node: GraphNode):
    out_vals = fx_node.meta.get('val')
    if out_vals is None:
      return

    if isinstance(out_vals, tuple):
      for idx, val in enumerate(out_vals):
        metadata = MetadataItem(id=str(idx), attrs=[])
        if val is None:
          continue
        dtype = str(val.dtype)
        shape = json.dumps(val.shape)
        metadata.attrs.append(
            KeyValue(key='tensor_shape', value=dtype + shape))
        node.outputsMetadata.append(metadata)
    else:
      dtype = str(out_vals.dtype)
      shape = json.dumps(out_vals.shape)
      metadata = MetadataItem(
          id='0', attrs=[KeyValue(key='tensor_shape', value=dtype + shape)]
      )
      node.outputsMetadata.append(metadata)

  def create_node(self, fx_node: torch.fx.node.Node):
    node = GraphNode(
        id=fx_node.name,
        label=self.get_label(fx_node),
        namespace=self.get_hierachy(fx_node),
    )
    self.add_incoming_edges(fx_node, node)
    self.add_node_attrs(fx_node, node)
    self.add_outputs_metadata(fx_node, node)
    return node

  def create_graph(self):
    graph = Graph(id='graph', nodes=[])
    for node in self.gm.graph.nodes:
      graph.nodes.append(self.create_node(node))
    return graph

  def convert(self) -> ModelExplorerGraphs:
    return {'graphs': [self.create_graph()]}
