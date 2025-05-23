import logging
from copy import deepcopy
from math import ceil, prod
from typing import Any

from rtree import index
from zigzag.datatypes import Constants, LayerDim, LayerOperand
from zigzag.utils import pickle_deepcopy

from stream.cost_model.group_allocation import GroupIdManager
from stream.hardware.architecture.accelerator import Accelerator
from stream.node_tensor import NodeTensor
from stream.opt.partitioning.TemporalLoop import TemporalLoop
from stream.opt.partitioning.utils import (
    convert_outer_cn_loops,
)
from stream.stages.stage import Stage, StageCallable
from stream.utils import contains_wildcard
from stream.workload.computation.computation_node import ComputationNode, LoopRanges
from stream.workload.dependency_propagation.concat_node import ConcatNode
from stream.workload.dependency_propagation.dummy_node import DummyNode
from stream.workload.dependency_propagation.elementwise_node import ElementwiseNode
from stream.workload.dependency_propagation.flatten_node import FlattenNode
from stream.workload.dependency_propagation.gather_node import GatherNode
from stream.workload.dependency_propagation.lpnormalization_node import LpNormalizationNode
from stream.workload.dependency_propagation.reshape_node import ReshapeNode
from stream.workload.dependency_propagation.transpose_node import TransposeNode
from stream.workload.dnn_workload import DNNWorkloadStream
from stream.workload.node import Node
from stream.workload.onnx_workload import ComputationNodeWorkload, ONNXWorkload
from stream.workload.tensor import SubviewTensor

logger = logging.getLogger(__name__)

EDGE_T = tuple[ComputationNode, ComputationNode, dict]


class TensorDimensionMismatchException(Exception):
    """Facilitates error handling in case incorrect tensor dimensions are passed on"""


class TiledWorkloadGenerationStage(Stage):
    """
    Class that transforms the layer-by-layer workload into finer CN workload graph.
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: ONNXWorkload,
        accelerator: Accelerator,
        **kwargs: Any,
    ):
        """
        Initialization of self.workload.
        :param main_inputs: MainInputs, NOT copied
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.accelerator = accelerator

        # Save for each of the workload's nodes the finer nodes that will be generated
        self.finer_nodes_dict: dict[ComputationNode, list[ComputationNode]] = {}

        # Memoize the numpy tensors for dependency generation
        self.numpy_tensors = {}

    def run(self):
        unique_finer_nodes: list[ComputationNode] = []
        # For each node get all the finer nodes and the edges between them
        all_finer_nodes = []
        all_finer_edges = []
        for node in self.workload.topological_sort():
            # If other node types shouldn't be included in finer node graph, add here
            if not isinstance(node, ComputationNode):
                continue
            outer_temporal_loops = self.get_outer_tmap_loop_dimensions(node)
            finer_nodes, unique_nodes = self.get_finer_nodes(node, outer_temporal_loops)
            logger.info(f"{node}: Outer loops {outer_temporal_loops}.")
            logger.info(f"{node}: Generated {len(finer_nodes)} finer nodes.")
            self.finer_nodes_dict[node] = finer_nodes
            unique_finer_nodes += unique_nodes
            intra_edges = self.get_intra_edges(finer_nodes)
            # Add the finer nodes and intra edges to the lists
            all_finer_nodes += finer_nodes
            all_finer_edges += intra_edges

        # Get all pairs of nodes that we have to extract inter edges for
        all_pairs = self.get_all_node_pairs(self.workload)
        for producer, consumer, is_complex in all_pairs:
            finer_producers = self.finer_nodes_dict[producer]
            finer_consumers = self.finer_nodes_dict[consumer]
            if is_complex:
                inter_edges = self.get_inter_edges_numpy(producer, consumer)
            else:
                inter_edges = self.get_inter_edges_rtree(producer, consumer, finer_producers, finer_consumers)
            all_finer_edges += inter_edges

        # Set the base_priority value of all nodes
        self.set_base_priority_of_nodes(all_finer_nodes, all_finer_edges)

        # Set the number of real predecessors of all nodes
        self.set_nb_real_predecessors(all_finer_nodes, all_finer_edges)

        # Construct the new finer workload graph
        # The graph construction needs to happen after the base priority and nb_real_predecessors are set
        partitioned_workload = ComputationNodeWorkload()
        partitioned_workload.add_nodes_from(all_finer_nodes)
        partitioned_workload.add_edges_from(all_finer_edges)

        logger.info(f"Finer graph: {partitioned_workload}.")

        kwargs = self.kwargs.copy()
        kwargs["original_workload"] = pickle_deepcopy(self.workload)
        kwargs["workload"] = partitioned_workload
        kwargs["accelerator"] = self.accelerator

        if "scheduling_order" not in kwargs:
            kwargs["scheduling_order"] = self.get_scheduling_order(partitioned_workload)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

        yield None, None

    @staticmethod
    def get_scheduling_order(workload: ComputationNodeWorkload):
        return sorted(((n.id, n.sub_id) for n in workload.node_list), reverse=True)

    @staticmethod
    def get_all_node_pairs(G: DNNWorkloadStream) -> tuple[tuple[ComputationNode, ComputationNode, bool], ...]:
        pairs: list[tuple[ComputationNode, ComputationNode, bool]] = []
        for node in G.topological_sort():
            if not isinstance(node, ComputationNode):
                continue
            successors = list(G.successors(node))
            is_computation_node = [isinstance(succ, ComputationNode) for succ in successors]
            while not all(is_computation_node):
                non_computation_node_succ_idx = is_computation_node.index(False)
                non_computation_node_succ = successors[non_computation_node_succ_idx]
                succ2 = list(G.successors(non_computation_node_succ))
                successors.pop(non_computation_node_succ_idx)
                successors += succ2
                is_computation_node = [isinstance(succ, ComputationNode) for succ in successors]

            # Now we have all ComputationNode successors
            for successor in successors:
                intermediates = G.shortest_path(node, successor)[1:-1]
                complex_pair = False
                for intermediate in intermediates:
                    if isinstance(intermediate, ComputationNode):
                        raise ValueError(
                            "Intermediate node between two ComputationNodes should not be a ComputationNode."
                        )
                    if not isinstance(intermediate, DummyNode):
                        complex_pair = True
                pairs.append((node, successor, complex_pair))
        return tuple(pairs)

    def get_outer_tmap_loop_dimensions(self, node: ComputationNode) -> list[TemporalLoop]:
        """Get the temporal loops that are outside a CN for this node.

        Args:
            node: node for which to return outer-cn loops

        Returns:
            temporal loops outside of cn
        """
        if contains_wildcard(node.inter_core_tiling):
            # inter core tiling is not set by CO yet
            tiling_to_split = node.intra_core_tiling
        else:
            # inter core tiling is ok, also split into these tiles
            tiling_to_split = node.intra_core_tiling + node.inter_core_tiling

        outer_loops = convert_outer_cn_loops(tiling_to_split, node)

        # In case no valid intra core tiling is found: add an arbitrary tiling of size 1
        if not outer_loops:
            outer_loops = [TemporalLoop(node.layer_dims[0], 1)]

        return outer_loops

    def get_non_type_predecessors(self, node: Node, types: list[type]) -> list[Node]:
        """Find all self.workload nodes that are not of any type in types.
        If a node of any type in types is a predecessor, we cascade back through the graph until only non-types type
        preds are found.

        Args:
            node (Node): the node for which we intend to find all preds that are not of a type in types
            types (list): list of different types that we want to exclude from our predecessors
        """
        preds: list[Node] = list(self.workload.predecessors(node))
        while any([type(pred) in types for pred in preds]):
            # Find first node in list that is of any type in types
            skip_node = next(pred for pred in preds if any([isinstance(pred, type) for type in types]))
            # Find its index
            idx = preds.index(skip_node)
            # Find its predecessors
            skip_node_preds = list(self.workload.predecessors(skip_node))
            # Pop the skip_node from the list of preds and append its preds to the list
            preds.pop(idx)
            preds += skip_node_preds
        return preds

    @staticmethod
    def get_finer_nodes(
        original_node: ComputationNode, outer_temporal_loops: list[TemporalLoop]
    ) -> tuple[list[ComputationNode], list[ComputationNode]]:
        original_node_id = original_node.id

        # Take away the outer_temporal_loops to create finer CNs for this node
        finer_node_attrs = original_node.extract_node_attr()
        finer_node_mapping = original_node.extract_inter_core_mapping_attr()

        for outer_tl in outer_temporal_loops:
            outer_dim = outer_tl.dimension
            outer_size = outer_tl.size
            # Check if this node's "dim" size is divisible by the outer-cn loop size
            node_dim_size = finer_node_attrs.layer_dim_sizes[outer_dim]
            q, rem = divmod(node_dim_size, outer_size)  # returns x//y, x%y
            if rem != 0:
                # Make sure that the outer_dim is divisible by the outer_size
                # Pad the dimension to a multiple of outer_size
                node_dim_size = q * outer_size
                q, rem = divmod(node_dim_size, outer_size)
                assert rem == 0, (
                    f"Node {original_node} dim {outer_dim} of size {node_dim_size} is not divisible by outer-cn temporal "
                    f"loop {outer_tl}"
                )
            finer_node_attrs.layer_dim_sizes[outer_dim] = q

        # Loop dimension + size of the finer nodes (called span here)
        finer_span = finer_node_attrs.layer_dim_sizes
        loop_dims = original_node.layer_dims
        stop_values = [temporal_loop.size for temporal_loop in outer_temporal_loops]
        nb_cns = int(prod(stop_values))

        # Compute the data_reuse_factor (will be used as base_priority later) for the constant operands of all CNs
        tensor_reuse_factors = deduce_tensor_reuse_factors(original_node, outer_temporal_loops)

        # Multiplication factor for each outer-cn loop.
        # This is to convert from the relative loop value which goes from 0, 1, ..., stop_value - 1
        # to the absolute value of that dimension (if there is another lower loop of the same type or spatial loop)
        mult_factors: list[int] = []
        for i, outer_loop in enumerate(outer_temporal_loops):
            loop_dim = outer_loop.dimension
            stop_value = outer_loop.size
            inner_span = finer_span[loop_dim] if loop_dim in finer_span else 1
            lower_outer_cn_loops = outer_temporal_loops[:i]
            # Returns 1 if empty list
            outer_span = prod(
                [temporal_loop.size for temporal_loop in lower_outer_cn_loops if temporal_loop.dimension == loop_dim]
            )
            mult_factors.append(int(inner_span * outer_span))

        finer_nodes: list[ComputationNode] = []
        tensors: list[SubviewTensor] = []
        output_tensor_range_to_final_producer: dict[tuple[int], ComputationNode] = {}
        group_id_manager = GroupIdManager(original_node)
        for n in range(nb_cns):
            outer_loop_values: list[int] = []
            for i, outer_loop in enumerate(outer_temporal_loops):
                loop_dim = outer_loop.dimension
                stop_value = outer_loop.size
                m = prod(stop_values[:i])
                outer_loop_values.append(int((n // m) % stop_value))
            dim_min_max: LoopRanges = {}
            for loop_dim in loop_dims:
                # find all outer-cn loops that iterate over this loop_dim
                # and multiply their loop values by their mult_factor
                dim_min = 0
                for i, outer_loop in enumerate(outer_temporal_loops):
                    dim = outer_loop.dimension
                    stop_value = outer_loop.size
                    if dim == loop_dim:
                        # current loop value of this outer-cn loop
                        loop_val = outer_loop_values[i]
                        # mult factor of this outer-cn loop
                        mult_factor = mult_factors[i]
                        dim_min += loop_val * mult_factor
                # max value is exclusive
                dim_max = dim_min + (finer_span[loop_dim] if loop_dim in finer_span else 1)
                dim_min_max[loop_dim] = (dim_min, dim_max)

            # finer_node_mapping_copy = deepcopy(original_node.extract_mapping_attr())
            group_id = group_id_manager.get_group_id(dim_min_max)

            # Create the computation node object with the computed ranges of the loop dimensions
            node_name = original_node.name
            # If all the output irrelevant loops are at a max, this is producing a final output, so set a flag
            original_node_output_ir_dims = original_node.loop_relevancy_info.get_ir_layer_dims(
                Constants.OUTPUT_LAYER_OP
            )

            produces_final_output = all(
                [dim_min_max[dim][1] >= original_node.layer_dim_sizes[dim] for dim in original_node_output_ir_dims]
            )

            # Get the operand tensors SubViewOps of the original node to build the SubViewTensors from
            original_subviews = {}
            for layer_op, tensor in original_node.operand_tensors.items():
                original_subviews[layer_op] = tensor.subview

            finer_node = ComputationNode(
                node_id=original_node_id,
                sub_id=n,
                node_name=node_name,
                node_attr=finer_node_attrs,
                mapping_attr=finer_node_mapping,
                op_type=original_node.type,
                produces_final_output=produces_final_output,
                group_id=group_id,
                subview_ops=original_subviews,
            )
            # Override loop_ranges property
            finer_node.update_loop_ranges(dim_min_max)
            # Re-calculate pr loop ranges based on new loop_ranges
            finer_node.calculate_pr_loop_ranges()
            # Re-set the operand tensors for the new loop_ranges
            finer_node.set_operand_tensors(original_subviews)

            # Initialize the priorities (total inter-CN data reuse factor) for the constant operands of this finer_node
            for constant_operand in finer_node.constant_operands:
                tensor = finer_node.operand_tensors[constant_operand]
                tensor.set_base_priorities(tensor_reuse_factors[constant_operand][n])

            # Replace any of the tensors with identical tensors of previous finer nodes
            for op, tensor in finer_node.operand_tensors.items():
                if op == Constants.OUTPUT_LAYER_OP:
                    continue
                replaced = False
                for previous_tensor in tensors:
                    if tensor.equality_hash() == previous_tensor.equality_hash():
                        finer_node.operand_tensors[op] = previous_tensor
                        replaced = True
                if not replaced:
                    tensors.append(tensor)

            output_tensor_loop_ranges = finer_node.operand_tensors[Constants.OUTPUT_LAYER_OP].loop_ranges
            output_tensor_range_to_final_producer[output_tensor_loop_ranges] = finer_node

            # Compute the output data produced by each finer node, assuming that all the data produced by different CNs
            # are unique
            finer_node.data_produced_unique = int(
                finer_node.operand_size_elem[Constants.OUTPUT_LAYER_OP]
                * finer_node.operand_precision[Constants.FINAL_OUTPUT_LAYER_OP]
            )

            # If the core allocation is fixed, we need to set the chosen core allocation.
            # It's possible the core allocation contains multiple entries.
            # In that case, we select the core allocation based on the group id.
            if original_node.core_allocation_is_fixed:
                assert group_id < len(
                    original_node.possible_core_allocation
                ), f"Group id {group_id} too large for core allocation list {original_node.core_allocation}"
                chosen_core_allocation = original_node.possible_core_allocation[group_id]
                finer_node.set_chosen_core_allocation(chosen_core_allocation)

            finer_nodes.append(finer_node)

        # Correct the output tensor of all CNs to the final producer (if multiple nodes handle the same output range)
        for node in finer_nodes:
            output_op = Constants.OUTPUT_LAYER_OP
            output_range = node.operand_tensors[output_op].loop_ranges
            final_producer = output_tensor_range_to_final_producer[output_range]
            node.operand_tensors[output_op] = final_producer.operand_tensors[output_op]

        # NOTE We take the first node as only unique one as they are all generated equally now.
        unique_finer_nodes = [finer_nodes[0]]

        return finer_nodes, unique_finer_nodes

    @staticmethod
    def get_intra_edges(nodes: list[ComputationNode]):
        # Get all the group ids
        group_ids = sorted(set([n.group for n in nodes]))
        intra_edges: list[tuple[ComputationNode, ComputationNode, dict[str, int]]] = []
        for group_id in group_ids:
            group_nodes = [n for n in nodes if n.group == group_id]
            pairs = zip(group_nodes, group_nodes[1:])
            for node_1, node_2 in pairs:
                intra_edges.append((node_1, node_2, {"bits": 0}))
        return intra_edges

    def convert_to_inclusive_data_range(self, exclusive_data_range: LoopRanges):
        """
        Convert an exclusive data range to an inclusive data range.
        """
        return {key: (min_val, max_val - 1) for key, (min_val, max_val) in exclusive_data_range.items()}

    def get_bounding_box_dimensions(
        self,
        producer: ComputationNode,
        consumer: ComputationNode,
        dimensions: list[LayerDim],
        loop_ranges: LoopRanges,
        interleaved: bool = True,
    ) -> tuple[int, ...]:
        """
        Extract the relevant dimension ranges for building the rtree with the dimensions in dimensions.
        The order of the operand's dimensions is determined through the dimensions parameter.
        """
        # Add compensation for grouped convolutions:
        # If there is a G dimension in the loop ranges alongside a C or K, it means we have a 5D tensor,
        # where the onnx tensors are always flattened back to 4D (merging the G+C or G+K into one channel dimension)
        dimensions, loop_ranges = self.flatten_grouped_convolution_ranges(producer, consumer, dimensions, loop_ranges)
        bounding_box = [loop_ranges[dim] for dim in dimensions]

        if not interleaved:
            bounding_box_flat = tuple([item for sublist in bounding_box for item in sublist])
            return bounding_box_flat
        else:
            bounding_box_flat = tuple(zip(*bounding_box))
            bounding_box_flat = tuple([item for sublist in bounding_box_flat for item in sublist])
            return bounding_box_flat

    def bounding_box_generator(
        self, producer: ComputationNode, consumer: ComputationNode, nodes: list[ComputationNode], operand: LayerOperand
    ):
        """
        Generator function that yields the bounding boxes of an operand for all nodes.
        """
        for i, node in enumerate(nodes):
            inclusive_ranges = self.convert_to_inclusive_data_range(node.loop_ranges)
            dimensions = node.operand_dimensionality_order[operand]
            bounds = self.get_bounding_box_dimensions(producer, consumer, dimensions, inclusive_ranges)
            yield (i, bounds, None)

    def get_nb_input_dimensions(self, node: ComputationNode, operand: LayerOperand):
        """Return the number of input dimensions this node has.
        # We take the first non-constant input operand."""
        dims = node.operand_dimensionality_order[operand]

        if LayerDim("G") in dims and (LayerDim("C") in dims or LayerDim("K") in dims):
            # because later the generator will merge them into a single channel dim
            return len(dims) - 1
        else:
            return len(dims)

    def build_rtree(
        self, producer: ComputationNode, consumer: ComputationNode, nodes: list[ComputationNode], operand: LayerOperand
    ):
        """
        Build an rtree data structure based on each node in 'nodes' for the relevant dimensions of operand.
        """
        props = index.Property()
        # We assume all nodes in 'nodes' have identical dimensions
        props.dimension = self.get_nb_input_dimensions(nodes[0], operand)

        rtree = index.Index(self.bounding_box_generator(producer, consumer, nodes, operand), properties=props)
        return rtree

    def flatten_grouped_convolution_ranges(
        self, producer: ComputationNode, consumer: ComputationNode, dims: list[LayerDim], ranges: LoopRanges
    ):
        """If both C/K and G are present in dimensions, flatten their loop ranges so the tensor is 4D.

        Args:
            dimensions (list): list of the different tensor dimensions
            loop_ranges (dict): dict of the loop ranges for the current node.
        """
        # TODO these should be constants
        dim_G = LayerDim("G")
        dim_C = LayerDim("C")
        dim_K = LayerDim("K")
        dim_CH = LayerDim("CH")

        dims_copy = deepcopy(dims)
        ranges_copy = deepcopy(ranges)
        assert all([dim in ranges_copy for dim in dims_copy])

        if dim_G in dims_copy and (dim_C in dims_copy or dim_K in dims_copy):
            G_idx = dims_copy.index(dim_G)
            if dim_C in dims_copy:
                is_consumer = True
                C_K_idx = dims_copy.index(dim_C)
            elif dim_K in dims_copy:
                C_K_idx = dims_copy.index(dim_K)
                is_consumer = False
            else:
                return dims_copy, ranges_copy
            # Replace the G + C/K into one dimension we call "CH" (name doesn't really matter)
            (G_min, G_max_incl) = ranges_copy[dim_G]
            (C_K_min, C_K_max_incl) = ranges_copy[dims_copy[C_K_idx]]
            CH_min = G_min + C_K_min
            original_node = consumer if is_consumer else producer
            CH_max_incl = G_max_incl * original_node.layer_dim_sizes[dims_copy[C_K_idx]] + C_K_max_incl
            ranges_copy[LayerDim("CH")] = (CH_min, CH_max_incl)

            # Remove the G + C/K from the original dimensions list and add CH in its place
            min_idx = min(G_idx, C_K_idx)

            dims_copy.remove(dim_G)
            second_dim = dim_C if is_consumer else dim_K
            dims_copy.remove(second_dim)
            # insert it in place of G or C/K, whichever came first
            dims_copy.insert(min_idx, dim_CH)

        assert all([dim in ranges_copy for dim in dims_copy])
        return dims_copy, ranges_copy

    def get_inter_edges_rtree(
        self,
        producer: ComputationNode,
        consumer: ComputationNode,
        finer_producers: list[ComputationNode],
        finer_consumers: list[ComputationNode],
    ):
        """Function that finds the edges between a producer and consumer node,
        more specifically their finer counterparts producer_finer and consumer_finer.
        A communication node is inserted between each producer and consumer node.

        Args:
            producer: the producer node
            consumer: the consumer node
            finer_producers: list of finer producer nodes
            finer_consumers: list of finer consumer nodes
        """
        # Check all the different input operands of the consumer node that stem from the producer node
        # The direct predecessor of an input operand might be a DummyNode so we need to propagate back
        dependent_input_operands: list[LayerOperand] = []
        for operand, parent_node_id in consumer.input_operand_source.items():
            parent_node = self.workload.get_node_with_id(parent_node_id)
            if parent_node == producer:
                dependent_input_operands.append(operand)
            elif not isinstance(parent_node, ComputationNode):
                # Propagate to the first parent CN
                non_dummy_parents = self.get_non_type_predecessors(parent_node, [DummyNode])
                if producer in non_dummy_parents:
                    dependent_input_operands.append(operand)

        # edges will hold the cns that are dependent on each other [(prod_cn, cons_cn), ...]
        edges: list[tuple[ComputationNode, ComputationNode, dict[str, Any]]] = []

        for input_operand in dependent_input_operands:
            # Build the tree of all finer consumer nodes for this operand
            consumer_tree = self.build_rtree(producer, consumer, finer_consumers, input_operand)

            # As long as we haven't iterated through all of the output's operand's irrelevant dimensions,
            # we shouldn't add an edge to the consumer layer's nodes, as this would create unnecessary graph complexity
            # Because we have the intra-edges between the nodes, and because the nodes irrelevant loops are
            # incrementing, we can make the graph simpler by just having one edge at the final irrelevant loop iteration
            # producer node. # Get the relevant (including partially relevant) and irrelevant dimensions of the producer
            # node's output
            producer_r_dims_output = producer.operand_dimensionality_order[Constants.OUTPUT_LAYER_OP]
            producer_ir_dims_output = producer.loop_relevancy_info.get_ir_layer_dims(Constants.OUTPUT_LAYER_OP)

            # Iterate through all the producer nodes and get the consumer nodes that require its outputs,
            # taking into account that we only want an edge if the producer's irrelevant loops are at a max
            for finer_producer in finer_producers:
                # Get the output irrelevant loop ranges and check if they are at least at the max
                ir_dims_not_at_max = [
                    finer_producer.loop_ranges[ir_dim][1] < producer.loop_ranges[ir_dim][1]
                    for ir_dim in producer_ir_dims_output
                ]
                if any(ir_dims_not_at_max):
                    continue  # to the next finer producer

                p_inclusive_ranges = self.convert_to_inclusive_data_range(finer_producer.loop_ranges)
                p_bounding_box = self.get_bounding_box_dimensions(
                    producer, consumer, producer_r_dims_output, p_inclusive_ranges
                )

                # Get the finer consumer node ids that intersect with this finer producer node
                intersecting_consumer_node_ids = consumer_tree.intersection(p_bounding_box)

                for intersecting_consumer_node_id in intersecting_consumer_node_ids:
                    intersecting_consumer = finer_consumers[intersecting_consumer_node_id]
                    # Create a new communication node that will reside between the producer and consumer node
                    edges += [
                        (
                            finer_producer,
                            intersecting_consumer,
                            {
                                "operand": input_operand,
                                "bits": finer_producer.data_produced_unique,
                            },
                        )
                    ]

        return edges

    def get_inter_edges_numpy(
        self,
        producer: ComputationNode,
        consumer: ComputationNode,
    ):
        numpy_tensors: dict[ComputationNode, dict[LayerOperand, NodeTensor]] = {}
        all_inter_edges: list[tuple[ComputationNode, ComputationNode, dict[str, Any]]] = []

        def get_tensor_cn_for_op(node: ComputationNode, dependent_operand: LayerOperand):
            """And update the known tensors of computation nodes"""
            if node in numpy_tensors:
                tensor_cns = numpy_tensors[node]
            else:
                finer_nodes = self.finer_nodes_dict[node]
                tensor_cns = self.get_tensor_cns(node, finer_nodes)
                # Store result for later use
                numpy_tensors[node] = tensor_cns
            tensor = tensor_cns[dependent_operand]
            return tensor

        paths_between = list(self.workload.all_simple_paths(producer, consumer))
        # Remove paths between that contain a ComputationNode as intermediate
        paths_between = [
            path for path in paths_between if not any(isinstance(node, ComputationNode) for node in path[1:-1])
        ]
        assert (
            len(paths_between) > 0
        ), "No paths between producer and consumer found without ComputationNode in intermediates."
        for path_between in paths_between:
            # First node in the path is a ComputationNode, of which we extract the output operand dependency tensor
            first_node = path_between[0]
            assert isinstance(first_node, ComputationNode), "First node in path should be ComputationNode"
            tensor = get_tensor_cn_for_op(first_node, dependent_operand=Constants.OUTPUT_LAYER_OP)

            # Propagate through intermediate, non-computation nodes
            for _, node in enumerate(path_between[1:-1], start=1):
                if isinstance(node, ComputationNode):
                    raise ValueError("Intermediate nodes should not be of type ComputationNode.")
                tensor = self.propagate_cn_production_for_non_cn(node, tensor)

            # Final node: Computation node
            final_node: ComputationNode = path_between[-1]  # type: ignore
            assert isinstance(final_node, ComputationNode), "Last node in path should be ComputationNode"

            # Find the operand for which this last node connects to its predecessor
            dependent_operand = next(
                op for op, dependent_node_id in final_node.input_operand_source.items() if dependent_node_id == node.id
            )

            # Error handling of shape mismatches in tensor propagation
            def get_final_tensor_alt_operand():
                """Error handling case 1: sources for `W` and `I` operand are swapped for this node
                -> try the other one"""
                try:
                    alt_operand = next(op for op in final_node.input_operand_source if op != dependent_operand)
                except StopIteration:
                    # No alt operand was found -> we're still in trouble
                    raise TensorDimensionMismatchException
                return get_tensor_cn_for_op(final_node, alt_operand)

            def get_shape_inferred_propagated_tensor(tensor: NodeTensor, final_tensor: NodeTensor):
                """Error handling case 2: dimensions of ComputationNode (`final_tensor`) were altered by stream
                (e.g. to be properly divisible) but this is not reflected in `ConcatNode` with constant shape.
                 -> manually fix shape"""
                if not any(isinstance(node, ConcatNode) for node in path_between[1:-1]):
                    raise TensorDimensionMismatchException(
                        "This function only solves the case of errors due to constant shapes in ConcatNode"
                    )

                target_shape = final_tensor.tensor_shape
                propagated_shape = tensor.tensor_shape
                extension_axis = next(i for i in range(len(target_shape)) if target_shape[i] != propagated_shape[i])
                extension_value = target_shape[extension_axis] - propagated_shape[extension_axis]
                if extension_value <= 0:
                    raise TensorDimensionMismatchException(
                        "Propagated shape cannot be larger than (extended) found shape"
                    )
                extension_shape = tuple(
                    val if i != extension_axis else extension_value for i, val in enumerate(target_shape)
                )
                return tensor.concat_with_empty(extension_shape, extension_axis, variable_input_first=False)

            try:  # Regular case
                final_tensor = get_tensor_cn_for_op(final_node, dependent_operand)
                inter_edges = self.get_inter_edges_tensor_based(tensor, final_tensor)
            except TensorDimensionMismatchException:
                try:  # Error case 1
                    final_tensor = get_final_tensor_alt_operand()
                    inter_edges = self.get_inter_edges_tensor_based(tensor, final_tensor)
                except TensorDimensionMismatchException:
                    try:  # Error case 2
                        final_tensor = get_tensor_cn_for_op(final_node, dependent_operand)
                        tensor = get_shape_inferred_propagated_tensor(tensor, final_tensor)
                        inter_edges = self.get_inter_edges_tensor_based(tensor, final_tensor)
                    except TensorDimensionMismatchException:
                        # Error case 1 and 2 combined
                        final_tensor = get_final_tensor_alt_operand()
                        tensor = get_shape_inferred_propagated_tensor(tensor, final_tensor)
                        inter_edges = self.get_inter_edges_tensor_based(tensor, final_tensor)

            for producer, cons in inter_edges:
                all_inter_edges.append(
                    (
                        producer,
                        cons,
                        {
                            "operand": dependent_operand,
                            "bits": producer.data_produced_unique,
                        },
                    )
                )
        return all_inter_edges

    def propagate_cn_production_for_non_cn(self, node: Node, input_tensor: NodeTensor) -> NodeTensor:
        match node:
            case ReshapeNode():
                return node.reshape_operand_tensor(input_tensor)
            case TransposeNode():
                return node.transpose(input_tensor)
            case LpNormalizationNode():
                return node.lpnormalization_operand_tensor(input_tensor)
            case FlattenNode():
                return node.flatten(input_tensor)
            case ElementwiseNode():
                return input_tensor.copy()
            case GatherNode():
                return node.gather_operand_tensor(input_tensor)
            case ConcatNode():
                return node.concat(input_tensor)
            case DummyNode():
                return input_tensor
            case _:
                raise NotImplementedError(f"Tensor propagation not implemented for node {node.name}.")

    @staticmethod
    def get_inter_edges_tensor_based(producer_output_tensor: NodeTensor, consumer_input_tensor: NodeTensor):
        """This method obtains the edges between a producer and consumer.
        This is done by iterating through all finer consumer nodes,
        for each consumer node we create a window and get all the producer nodes that produced this data window.

        Args:
            producer_output_tensor (np.ndarray): A tensor containing for each position which CNs will produce it
            consumer_input_tensor (np.ndarray): A tensor containing for each position which CNs will consume it
        """
        if producer_output_tensor.tensor_shape != consumer_input_tensor.tensor_shape:
            raise TensorDimensionMismatchException("Arrays to construct inter-layer edges must be equal shape.")

        inter_edges: set[tuple[ComputationNode, ComputationNode]] = set()
        for producer_array, consumer_array in zip(producer_output_tensor.flat, consumer_input_tensor.flat):
            for producer in producer_array:
                # The producer/consumer array may contain a lot of 0
                if not producer:
                    continue
                for consumer in consumer_array:
                    if not consumer:
                        continue

                    inter_edges.add((producer, consumer))
        return inter_edges

    def get_tensor_cns(
        self, node: ComputationNode, finer_nodes: list[ComputationNode]
    ) -> dict[LayerOperand, NodeTensor]:
        is_source_node = len(self.get_non_type_predecessors(node, [DummyNode])) == 0
        variable_operands = [op for op in node.input_operands if op not in node.constant_operands] + [
            node.output_operand
        ]
        tensor_dims = {op: node.operand_dimensionality_order[op] for op in variable_operands}
        all_loop_dim_sizes = node.layer_dim_sizes + node.pr_layer_dim_sizes  # union
        tensor_shapes = {op: tuple(all_loop_dim_sizes[dim] for dim in dims) for (op, dims) in tensor_dims.items()}

        # Initial arrays.
        tensors_cns: dict[LayerOperand, NodeTensor] = {
            op: NodeTensor.initialize_empty(shape) for (op, shape) in tensor_shapes.items()
        }

        # For each input operand iterate through the finer_nodes in reverse order
        # because we want the first cn with a dependency saved in the tensor
        # For the output operand iterate through the finer_nodes in regular order
        # because we want the last CN that handles an output tensor window to be saved
        for op, dims in tensor_dims.items():
            if op == node.output_operand:
                ir_dims_output = node.loop_relevancy_info.get_ir_layer_dims(Constants.OUTPUT_LAYER_OP)
                finer_nodes_list = finer_nodes  # list in regular order
                should_add_to_tensor_list = [
                    all(finer_node.loop_ranges[ir_dim][1] >= node.loop_ranges[ir_dim][1] for ir_dim in ir_dims_output)
                    for finer_node in finer_nodes_list
                ]
                attr_to_add_to = "data_produced_unique"
                precision = node.operand_precision[Constants.FINAL_OUTPUT_LAYER_OP]
            else:
                finer_nodes_list = list(reversed(finer_nodes))  # list in reversed order
                should_add_to_tensor_list = [True for _ in finer_nodes_list]
                attr_to_add_to = "data_consumed_unique"
                # if this layer is the first layer, we assume the inputs are streamed and "free"
                precision = node.operand_precision[op] * (not is_source_node)

            nb_unique_data_seen = 0
            for finer_node, should_add_to_tensor in zip(finer_nodes_list, should_add_to_tensor_list):
                if not should_add_to_tensor:
                    continue  # Skip if we're not at the max ir loop value for output
                op_dim_ranges = [finer_node.loop_ranges[loop_dim] for loop_dim in dims]
                op_dim_ranges_max_stop = tuple(tensor_shapes[op])
                # start can be negative for padding which, makes np flip
                window = tuple([slice(max(0, start), stop) for (start, stop) in op_dim_ranges])
                # Count how many nans we have in this window, as this is the amount of unique data consumed/produced by
                # this finer_node
                nb_unique_data_bits = tensors_cns[op].get_nb_empty_elements(window) * precision
                nb_unique_data_seen += nb_unique_data_bits
                # Add this amount of unique data to the "data_consumed_unique" or "data_produced_unique" depending on
                # input/output operand
                setattr(
                    finer_node,
                    attr_to_add_to,
                    getattr(finer_node, attr_to_add_to) + nb_unique_data_bits,
                )
                # Set this window of the tensor to indicate it will be consumed/produced by this finer node
                bounded_op_dim_ranges = tuple(
                    slice(max(0, start), min(max_stop, stop))
                    for ((start, stop), max_stop) in zip(op_dim_ranges, op_dim_ranges_max_stop)
                )
                tensors_cns[op] = tensors_cns[op].extend_with_node(bounded_op_dim_ranges, finer_node)

            if nb_unique_data_seen != (prod(tensor_shapes[op]) * precision):
                logger.warning(f"Downsampling node detected: {node}, operand= {op}.")

        # The dimensionality order of this input/output operand might include
        # both a G and C/K dimension because the ComputationNode gets the group as an extra
        # dimension in its input/output operand to have a notion of the "grouped" concept.
        # Here we reduce the input/output tensor from 5D to 4D tensor for such cases, e.g.:
        # input operand with dimensionality_order = ['B', 'G', 'C', 'IY', 'IX']
        #   -> gets reduced to dimensionality_order = ['B', 'CH', 'IY', 'IX']
        #       (in this case the 'CH' represents the absolute "channel" dimension)
        for op, tensor in tensors_cns.items():
            tensors_cns[op] = node.reshape_operand_tensor(tensor, operand=op)

        return tensors_cns

    @staticmethod
    def set_base_priority_of_nodes(nodes: list[ComputationNode], edges: list[EDGE_T]):
        """Set the base_priority of all stored tensors of variable operands in every node in finer_nodes
        based on the amount of real (excluding same layer edges) edges.

        Args:
            nodes (list): List of nodes.
            edges (list): List of edges in the form of (producer, consumer, data).
        """
        for node in nodes:
            output_operand = node.output_operand
            output_tensor = node.operand_tensors[output_operand]
            successors = [cons for prod, cons, _ in edges if prod == node]
            output_tensor.set_base_priorities(len(successors))

    @staticmethod
    def set_nb_real_predecessors(nodes: list[ComputationNode], edges: list[EDGE_T]):
        """Set the number of real predecessors for each node in the graph.
        A real predecessor is a node that is not in the same layer as the node itself.
        """
        for node in nodes:
            nb_real_predecessors = [prod for prod, cons, _ in edges if cons == node and prod.id != cons.id]
            node.nb_real_predecessors = len(nb_real_predecessors)

    def get_weight_capacities(self):
        # Get the weight capacity of all cores
        weight_capacities: dict[int, int] = {}
        for core in self.accelerator.cores.node_list:
            if core.id == self.accelerator.offchip_core_id:
                continue  # skip offchip core
            core_weight_capacity = core.memory_hierarchy.get_operand_top_level(Constants.MEM_OP_2).memory_instance.size
            weight_capacities[core.id] = core_weight_capacity
        return weight_capacities

    def get_layer_split_factors_k(self):
        # Get for each layer the split factor we need to be able to fit weights on possible cores
        split_factors: dict[ComputationNode, int] = {}
        for node in self.workload.node_list:
            if isinstance(node, DummyNode):
                continue
            # Get the weight capacity of all possible core allocations of this node
            core_allocations = node.possible_core_allocation
            # for fixed single allocation don't consider the splitting
            if len(core_allocations) == 1:
                continue
            core_capacities = [self.weight_capacities[core_id] for core_id in core_allocations]
            min_core_capacity = min(core_capacities)
            # Get the weight size of this layer
            constant_operands = node.constant_operands
            if not constant_operands:
                continue

            constant_operand = node.constant_operands[0]
            weight_size = node.operand_size_bit[constant_operand]
            if weight_size == 0:
                continue
            split_factor = ceil(weight_size / (self.split_W_percentage * min_core_capacity))  # 0.5 for double buffering
            if split_factor == 1:
                continue
            # Check if the split_factor is a divisor of the number of output channels
            try:
                output_channels = node.layer_dim_sizes[LayerDim("K")]
            except KeyError:
                raise NotImplementedError(f"{node} doesn't have a 'K' loop.")
            while divmod(output_channels, split_factor)[1] != 0:
                split_factor += 1
                if split_factor > output_channels:
                    raise ValueError("Something went wrong.")
            split_factors[node] = split_factor
        return split_factors


def deduce_tensor_reuse_factors(
    original_node: ComputationNode, outer_temporal_loops: list[TemporalLoop]
) -> dict[LayerOperand, list[int]]:
    """This function is used to generate a list of inter-CN data reuse factor for each CN's constant operand, like W,
      based on the outer-CN loops and the r, ir relations.

    Args:
        original_node (ComputationNode): the original layer node before tilling
        outer_temporal_loops (list[TemporalLoop]): the outer CN temporal loops

    Returns:
        data_reuse_factor (dict[list[int]]): a list of data reuse factor (base priority) for constant operands of each
        CN
    """

    # If there is no loop in the r_ir_loop, meaning that there is no outer-CN loop -> layer-by-layer
    if not outer_temporal_loops:
        return {}

    if not original_node.constant_operands:
        return {}

    # Transfer the outer_temporal_loops to r_ir_loop.
    #  An example can be r_ir_loop = {'W': [('ir', 3), ('r', 2), ('ir', 3)]}.
    r_ir_LUT = original_node.loop_relevancy_info
    constant_operands = original_node.constant_operands
    r_ir_loop: dict[LayerOperand, list[tuple[str, int]]] = {}
    for constant_operand in constant_operands:
        r_ir_loop[constant_operand] = []
        for loop in outer_temporal_loops:
            if loop.dimension in r_ir_LUT.get_ir_layer_dims(constant_operand):
                r_ir_loop[constant_operand].append(("ir", loop.size))
            else:
                r_ir_loop[constant_operand].append(("r", loop.size))

    # total_reuse_factor is the upper bound of the reuse factor that current layer CNs can reach
    total_reuse_factors = {
        op: prod([reuse_factor for (loop_type, reuse_factor) in r_ir_loop[op] if loop_type == "ir"])
        for op in r_ir_loop.keys()
    }

    # total number of nodes that will be generated
    nb_nodes = prod([tl.size for tl in outer_temporal_loops])

    # tensor reuse factor will be set to the total reuse factor for each node
    # whenveer a cn will be scheduled, the tensor reuse factor will decrease
    tensor_reuse_factors: dict[LayerOperand, list[int]] = {}
    for op, total_reuse_factor in total_reuse_factors.items():
        tensor_reuse_factors[op] = [total_reuse_factor] * nb_nodes

    return tensor_reuse_factors
