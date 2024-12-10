# Copyright 2024 The AI Edge Model Explorer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Literal, Union

from .utils import remove_none

Num = Union[float, int]
NodeData = Union['ModelNodeData', 'GraphNodeData']


# The node data for a model.
#
# It stores a dict that indexes `GraphNodeData` by graph ids.
@dataclass
class ModelNodeData:
  """The node data for a model."""

  # Graph node data indexed by graph ids.
  graphsData: dict[str, 'GraphNodeData']

  def save_to_file(self, path: str, indent: Union[int, None] = 2) -> None:
    """Writes the data into the given file."""
    abs_path = os.path.abspath(os.path.expanduser(path))
    with open(abs_path, 'w') as f:
      data = self.to_json_string(indent=indent)
      f.write(data)

  def to_json_string(self, indent: Union[int, None] = None) -> str:
    data = {k: remove_none(asdict(v)) for (k, v) in self.graphsData.items()}
    return json.dumps(data, indent=indent)


@dataclass
class GraphNodeData:
  """Node data for a single graph."""

  # Node data values indexed by node keys.
  #
  # The key could be:
  # - Any of the output tensor names of a node. It will be used to match a
  #   node's `tensor_name` key in its `outputsMetadata`. See graph_builder.py
  #   for more info.
  # - The node id specified in `GraphNode`. See graph_builder.py for more info.
  results: dict[str, 'NodeDataResult']

  # Thresholds that define various ranges and the corresponding node styles
  # (e.g. node bg color) to be applied for that range.
  #
  # Take the following thresholds as an example:
  #
  # [
  #   {value: 10, bgColor: 'red'}
  #   {value: 50, bgColor: 'blue'}
  #   {value: 100, bgColor: 'yellow'}
  # ]
  #
  # This means:
  # - Node data with value <=10 have "red" background color.
  # - Node data with value >10 and <=50 have "blue" background color.
  # - Node data with value >50 and <=100 have "yellow" background color.
  # - Node data with value >100 have no background color (white).
  #
  # (optional)
  thresholds: list['ThresholdItem'] = field(default_factory=list)

  # A gradient that defines the stops (from 0 to 1) and the associated colors.
  # A stop value 0 corresponds to the minimum value in `results`, and a stop
  # value 1 corresponds to the maximum value in results. Stops for 0 and 1
  # should always be provided.
  #
  # When color-coding a node, the system uses the node's data value to
  # calculate a corresponding stop and interpolates between gradient colors.
  #
  # This field takes precedence over the `thresholds` field above.
  #
  # (optional)
  gradient: list['GradientItem'] = field(default_factory=list)

  # Whether to hide the corresponding column in aggregated stats table
  # (the first table).
  #
  # If all columns in that table are hidden, the whole table will be hidden.
  #
  # (optional)
  hideInAggregatedStatsTable: bool = False

  # Whether to hide the corresponding column in children stats table
  # (the second table).
  #
  # If all columns in that table are hidden, the whole table will be hidden.
  #
  # (optional)
  hideInChildrenStatsTable: bool = False

  # The stats to hide in the aggregated stats table (the first table).
  #
  # The value for the hidden stat will be displayed as '-'.
  #
  # (optional)
  hideAggregatedStats: Union[None, list['AggregatedStat']] = None

  # Controls whether to display a detailed value distribution summary on the
  # group node.
  #
  # By default, a color bar representing the value distribution of
  # all descendant nodes is shown at the bottom of the group node. If this
  # field is set to true, we will show a more detailed summary, with each
  # value's label, percentage, and count shown on a separate line.
  #
  # For now this only works with non-numerical (e.g. string) node data values.
  #
  # (optional)
  showExpandedSummaryOnGroupNode: bool = False

  # Whether to display the label count columns in the children stats table in
  # the side panel.
  #
  # For now this only works with non-numerical (e.g. string) node data values.
  #
  # (optional)
  showLabelCountColumnsInChildrenStatsTable: bool = False

  def save_to_file(self, path: str, indent: Union[int, None] = 2) -> None:
    """Writes the data into the given file."""
    abs_path = os.path.abspath(os.path.expanduser(path))
    with open(abs_path, 'w') as f:
      data = self.to_json_string(indent=indent)
      f.write(data)

  def to_json_string(self, indent: Union[int, None] = None) -> str:
    data = remove_none(asdict(self))
    return json.dumps(data, indent=indent)


@dataclass
class NodeDataResult:
  """A single result data corresponding to a node."""

  # A numeric value.
  value: Num

  # The background color to render the corresponding node with.
  #
  # This could be any CSS color. It takes precedence over the color calculated
  # from `GraphNodeData.thresholds` and `GraphNodeData.gradient`.
  #
  # (optional)
  bgColor: Union[str, None] = None

  # The text label color to render the corresponding node with.
  #
  # This could be any CSS color. It takes precedence over the color calculated
  # from `GraphNodeData.thresholds` and `GraphNodeData.gradient`.
  #
  # (optional)
  textColor: Union[str, None] = None


@dataclass
class ThresholdItem:
  """A threshold item with the upperbound value and its corresponding colors."""

  # The value of the threshold.
  value: Num

  # The background color to render for node whose value falls into the segment
  # defined by this threshold item.
  #
  # This could be any CSS color.
  bgColor: str

  # The text label color to render for node whose value falls into the segment
  # defined by this threshold item.
  #
  # This could be any CSS color.
  #
  # (optional)
  textColor: Union[str, None] = None


@dataclass
class GradientItem:
  """A gradient item with the stop and its corresponding colors."""

  # A number from 0 to 1.
  #
  # 0 and 1 corresponds to the minimum/maximum value of the all the values in
  # data.
  stop: Num

  # The background color to render at this stop.
  #
  # Only support hex color (e.g. #aabb00) or color names (e.g. red).
  #
  # (optional)
  bgColor: Union[str, None] = None

  # The text color to render at this stop.
  #
  # Only support hex color (e.g. #aabb00) or color names (e.g. red).
  #
  # (optional)
  textColor: Union[str, None] = None


AggregatedStat = Literal['min', 'max', 'sum', 'avg']
