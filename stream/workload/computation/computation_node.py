from copy import deepcopy
from typing import TypeAlias

from xdsl.dialects.builtin import MemRefType, i8, i16, i32
from xdsl.dialects.memref import AllocOp, SubviewOp
from zigzag.datatypes import Constants, LayerDim, LayerOperand, MemoryOperand
from zigzag.utils import hash_sha512
from zigzag.visualization.results.plot_cme import shorten_onnx_layer_name
from zigzag.workload.layer_attributes import (
    LayerPadding,
)
from zigzag.workload.layer_node import LayerNode, LayerNodeAttributes

from stream.node_tensor import NodeTensor
from stream.workload.mapping import INTRA_CORE_MAPPING_DEFAULT, InterCoreMappingAttributes
from stream.workload.node import Node
from stream.workload.tensor import SubviewTensor

OperandTensorReshape: TypeAlias = dict[LayerOperand, tuple[int, ...]]
LoopRanges: TypeAlias = dict[LayerDim, tuple[int, int]]

PRECISION_TYPE_MAP = {
    8: i8,
    16: i16,
    32: i32,
}


class ComputationNode(LayerNode, Node):
    """Extension of ZigZag's concept of a "LayerNode" into a more general concept
    called "ComputationNode", which is not necessarily an entire layer,
    but can represent a smaller chunk of a layer.
    This object also inherits from the "Node" class, which is an abstract baseclass to represent
    different types of onnx nodes needed to accurately schedule the fine-grained graph.
    On top of that, some new information is added for correct dependency generation
    for the finer graph that is built when a layer is split into one and is a
    producer/consumer of another layer.
    """

    too_large_operands: list[MemoryOperand]

    # Map the node's op_type to the corresponding layer dimension to split on for fusion
    FUSION_DIM_MAPPING: dict[str, list[LayerDim]] = {
        "conv": [LayerDim("OY")],
        "matmul": [LayerDim("D")],
        "gemm": [LayerDim("D")],
        "pooling": [LayerDim("OY")],
        "add": [LayerDim("D")],
        "mul": [LayerDim("D")],
        "softmax": [LayerDim("K")],
        "max": [LayerDim("K")],
        "div": [LayerDim("K")],
        "exp": [LayerDim("K")],
        "sum": [LayerDim("K")],
        "relu": [LayerDim("K")],
        "gelu": [LayerDim("K")],
        "silu": [LayerDim("K")],
    }  # TODO default to "K" ?

    def __init__(
        self,
        node_id: int,
        node_name: str,
        node_attr: LayerNodeAttributes,
        mapping_attr: InterCoreMappingAttributes,
        op_type: str = "computation",
        operand_tensor_reshape: OperandTensorReshape | None = None,
        produces_final_output: bool = False,
        group_id: int = 0,
        sub_id: int = -1,  # To distinguish alternative versions of this node
        subview_ops: dict[SubviewOp] = {},
    ):
        op_type = op_type.lower()

        LayerNode.__init__(
            self, layer_id=node_id, node_name=node_name, node_attr=node_attr, mapping_attr=INTRA_CORE_MAPPING_DEFAULT
        )
        Node.__init__(
            self,
            node_id=node_id,
            node_name=node_name,
            type=op_type,
            onchip_energy=0,
            offchip_energy=0,
            runtime=0,
            possible_core_allocation=mapping_attr.core_allocation,
        )

        # Overwrite default spatial mapping with given one
        self.spatial_mapping = mapping_attr.spatial_mapping
        # Unpack other mapping attributes
        self.core_allocation = mapping_attr.core_allocation
        self.core_allocation_is_fixed = mapping_attr.core_allocation_is_fixed
        self.intra_core_tiling = mapping_attr.intra_core_tiling
        self.inter_core_tiling = mapping_attr.inter_core_tiling
        self.kernel = mapping_attr.kernel

        self.sub_id = sub_id
        self.group = group_id
        self.operand_tensor_reshape = (
            operand_tensor_reshape if operand_tensor_reshape is not None else self.get_operand_tensor_reshape_default()
        )
        self.produces_final_output = produces_final_output
        self.loop_ranges: LoopRanges = {  # type: ignore
            layer_dim: (0, size) for layer_dim, size in self.layer_dim_sizes.items()
        }
        self.operand_dimensionality_order: dict[LayerOperand, list[LayerDim]] = {
            layer_op: self.equation.get_r_layer_dims(layer_op) for layer_op in self.equation.get_contained_operands()
        }

        # adds pr dimensions loop ranges to self.loop_ranges
        self.calculate_pr_loop_ranges()
        # Rename function
        self.get_node_operand = self.memory_operand_links.mem_to_layer_op
        self.extract_node_info = self.extract_layer_info

        # Number of real predecessors is saved to deal with edge cases where some nodes of the same layer have differing predecessors
        # This is used to hash the node and to get accurate knowledge of the number of unique nodes.
        # This should be set after the node is created and the number of predecessors is known.
        self.nb_real_predecessors = None
        self._static_hash_value = self.__compute_static_hash()

        try:
            self.fusion_partition_dims = ComputationNode.FUSION_DIM_MAPPING[op_type]
        except KeyError:
            raise NotImplementedError(f"Fusion partitioning dimensions not defined for {op_type}")

        # Each ComputationNode will save a tensor for all its defined operands.
        # For example, a conv layer will have an I tensor, W tensor and O tensor.
        self.operand_tensors: dict[LayerOperand, SubviewTensor] = {}
        self.set_operand_tensors(subview_ops)

    def set_operand_tensors(self, subview_ops: dict[SubviewOp] | None):
        """Set the operand tensors for this node based on the given subview ops."""
        for op in self.layer_operands:
            if op == Constants.OUTPUT_LAYER_OP:
                precision = self.operand_precision.final_output_precision
            else:
                precision = self.operand_precision[op]

            op_dimensionality_order = self.operand_dimensionality_order[op]
            ranges = tuple([self.loop_ranges[dim] for dim in op_dimensionality_order])
            offsets = [x[0] for x in ranges]
            sizes = [x[1] - x[0] for x in ranges]
            strides = [1] * len(ranges)
            precision_type = PRECISION_TYPE_MAP.get(precision)
            memref_type = MemRefType(precision_type, sizes)
            if subview_ops:
                assert op in subview_ops, f"SubviewOp not found for operand {op}"
                memref_source = subview_ops[op]
            else:
                memref_source = AllocOp([], [], memref_type)
            self.operand_tensors[op] = SubviewTensor(
                memref_source=memref_source,
                memref_type=memref_type,
                offsets=offsets,
                sizes=sizes,
                strides=strides,
                cn_source=self,
                layer_operand=op,
                loop_dimensions=op_dimensionality_order,
                loop_ranges=ranges,
            )

    def get_operand_tensor_reshape_default(self) -> OperandTensorReshape | None:
        try:
            size_B = self.layer_dim_sizes[LayerDim("B")]
            size_OX = self.layer_dim_sizes[LayerDim("OX")]
            size_OY = self.layer_dim_sizes[LayerDim("OY")]
            size_IX = self.pr_layer_dim_sizes[LayerDim("IX")]
            size_IY = self.pr_layer_dim_sizes[LayerDim("IY")]
            return {
                LayerOperand("I"): (size_B, -1, size_IX, size_IY),
                LayerOperand("O"): (size_B, -1, size_OX, size_OY),
            }
        except KeyError:
            return None

    @property
    def short_name(self) -> str:
        return shorten_onnx_layer_name(self.name)

    def __compute_static_hash(self):
        """Return a value that can be used to identify unique nodes in sets, dicts and equality. It is pre-computed at
        initialization time to speed up dict lookup and instance equality"""
        return hash_sha512(
            (
                self.layer_dim_sizes,
                frozenset(self.dimension_relations),
                self.operand_precision,
                self.memory_operand_links,
                self.id,
                self.sub_id,
                self.nb_real_predecessors,
            )
        )

    def __str__(self):
        return f"ComputationNode{self.id}_{self.sub_id}"

    def __repr__(self):
        return str(self)

    def __hash__(self) -> int:
        """The hash operator of a node.

        Returns:
            the pre-computed hash
        """
        return self._static_hash_value

    def __eq__(self, other: object):
        """Fast equality comparison between two nodes"""
        # Optimization: this method is used many times to compare with `0`, to count empty tensor elements
        if not other:
            return False
        return isinstance(other, ComputationNode) and self._static_hash_value == other._static_hash_value

    def has_same_performance(self, other: object) -> bool:
        """Compare the equality between two nodes.
        Two nodes are considered equal if they have equal hardware performance, which happens following attributes are
        equal:
        - loop_dim_size: The size of the loops.
        - dimension_relations: The partial relevancy between a dimension and two others.
        - operand_precision: The precision at which the operands are stored, which means the operand identifiers should
          be equal.
        - memory_operand_links: The link between memory operand (paths in mem hierarchy) and this node's operands
          accurate knowledge of the number of unique nodes.
        - nb_real_predecessors: The number of predecessors of the node. This impacts the required memory size.

        Args:
            other (Node): The other node to compare this node with

        Returns:
            bool: Whether the nodes are equal or not
        """
        return (
            isinstance(other, ComputationNode)
            and self.layer_dim_sizes == other.layer_dim_sizes
            and self.dimension_relations == other.dimension_relations
            and self.operand_precision == other.operand_precision
            and self.memory_operand_links == other.memory_operand_links
            and self.id == other.id
            and self.nb_real_predecessors == other.nb_real_predecessors
            # NOTE: don't include sub_id
        )

    def __lt__(self, other: "ComputationNode"):
        """Compare two ComputationNodes for the 'less than (<)' operator.

        Args:
            other (ComputationNode): The other ComputationNode.

        Returns:
            bool: self < other
        """
        return (self.id, self.sub_id) < (other.id, other.sub_id)

    def get_operand_for_dim(self, dim: LayerDim) -> LayerOperand:
        """Return the first operand in the operand_list that has this dim as one of is dimensions

        Args:
            dim (str): The dimension for which to find the operand

        Returns:
            str: The operand that has dim as one of its dimensions
        """
        for op in self.layer_operands:
            if dim in self.operand_dimensionality_order[op]:
                return op
        raise ValueError(f"The given dim {dim} doesn't appear in any operand's dimensionality order")

    def calculate_pr_loop_ranges(self):
        """Add the loop ranges of the partially revelant dimensions for this node to self.loop_ranges"""
        for pr_dim, related_dims_and_scalings in self.pr_scaling_factors.items():
            dim_padding = self.padding[pr_dim] if pr_dim in self.padding else LayerPadding.DEFAULT
            padding_begin = dim_padding[0]
            # Assume that there is always 2 dimensions involved in the calculation of a pr dimension
            pr_dim_val_min = -padding_begin
            pr_dim_val_max = -padding_begin
            for related_dimension, scaling_factor in related_dims_and_scalings:
                pr_dim_val_min += scaling_factor * self.loop_ranges[related_dimension][0]
                # convert to inclusive upper limit
                pr_dim_val_max += scaling_factor * (self.loop_ranges[related_dimension][1] - 1)
            pr_dim_val_max += 1  # convert to exclusive upper range
            self.loop_ranges[pr_dim] = (pr_dim_val_min, pr_dim_val_max)

    def reshape_operand_tensor(self, tensor: NodeTensor, operand: LayerOperand):
        """Reshape the tensor back to the representation needed for producer/consumer."""
        if self.operand_tensor_reshape is None or operand not in self.operand_tensor_reshape:
            return tensor
        else:
            new_shape = self.operand_tensor_reshape[operand]
            return tensor.reshape(new_shape)

    def set_too_large_operands(self, too_large_operands: list[MemoryOperand]):
        self.too_large_operands = too_large_operands

    def update_loop_ranges(self, new_ranges: LoopRanges):
        """Override the loop ranges with a new value for each of the given LayerDims. Keep the old range for the
        LayerDims not defined in `new_ranges`"""
        for layer_dim in new_ranges:
            self.loop_ranges[layer_dim] = new_ranges[layer_dim]

    def extract_inter_core_mapping_attr(self):
        mapping_attr = InterCoreMappingAttributes(
            op_type=self.type,
            spatial_mapping=self.spatial_mapping,
            core_allocation=self.core_allocation,
            core_allocation_is_fixed=self.core_allocation_is_fixed,
            intra_core_tiling=self.intra_core_tiling,
            inter_core_tiling=self.inter_core_tiling,
            kernel=self.kernel,
        )
        return deepcopy(mapping_attr)

    @property
    def nb_real_predecessors(self):
        return self.__nb_real_predecessors

    @nb_real_predecessors.setter
    def nb_real_predecessors(self, nb_real_predecessors: int | None):
        self.__nb_real_predecessors = nb_real_predecessors
        self._static_hash_value = self.__compute_static_hash()
