from math import prod
from typing import TYPE_CHECKING, Sequence

from xdsl.dialects.builtin import MemRefType
from xdsl.dialects.memref import AllocOp, SubviewOp
from zigzag.datatypes import LayerDim, LayerOperand

if TYPE_CHECKING:
    from zigzag.hardware.architecture.memory_instance import MemoryInstance

    from stream.cost_model.memory_manager import MemoryManager
    from stream.hardware.architecture.accelerator import Accelerator
    from stream.workload.computation.computation_node import ComputationNode
    from stream.workload.onnx_workload import ComputationNodeWorkload


class SubviewTensor:
    """Class to represent a data tensor.
    TODO: Add from which layer this tensor originates and its dimension ranges
    """

    def __init__(
        self,
        memref_source: AllocOp,
        memref_type: MemRefType,
        offsets: Sequence[int],
        sizes: Sequence[int],
        strides: Sequence[int],
        cn_source: "ComputationNode",
        layer_operand: LayerOperand,
        loop_dimensions: list[LayerDim],
        loop_ranges: tuple[tuple[int, int], ...],
    ):
        """Initialize the Tensor instance.

        Args:
            size (int): the size of the tensor in bits
            origin (ComputationNode): The computation node that consumes/produces this tensor
            layer_operand (str, optional): The layer operand to which this tensor belongs
            loop_dimensions (tuple, optional): The loop dimensions for this tensor
            loop_ranges (tuple, optional): The loop range span for the different dimensions of this operand
        """
        subview = SubviewOp.from_static_parameters(
            source=memref_source,
            source_type=memref_type,
            offsets=offsets,
            sizes=sizes,
            strides=strides,
        )
        self.subview = subview
        self.size = prod(sizes)
        self.cn_source = cn_source
        self.layer_operand = layer_operand
        self.memory_operand = self.cn_source.memory_operand_links.layer_to_mem_op(layer_operand)
        self.loop_dimensions = loop_dimensions
        self.loop_ranges = loop_ranges
        self.base_priority: None | int = None  # Will be set when we know how many successors this node has (static)
        self.instance_priorities: dict[MemoryInstance, int] = {}
        self.id = (self.cn_source.id, self.cn_source.sub_id, layer_operand)

    def __str__(self) -> str:
        return f"Tensor{self.id}"

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return self.size

    def __hash__(self) -> int:
        return hash((self.cn_source, self.layer_operand))

    def __lt__(self, __o: object) -> bool:
        return isinstance(__o, SubviewTensor) and self.size < __o.size

    def equality_hash(self):
        return hash((self.cn_source.id, self.layer_operand, self.loop_ranges))

    def set_base_priorities(self, base_priority: int):
        self.base_priority = base_priority

    def get_instance_priority(self, top_instance: "MemoryInstance", memory_manager: "MemoryManager"):
        if top_instance in self.instance_priorities:
            return self.instance_priorities[top_instance]
        else:
            # If the top_instance is not in the dict. it means the core_id is the core that generates the tensor.
            # We  then return as priority the sum of all priorities of top instances that are not sotring the tensor.
            storing_instances, _, _ = memory_manager.find_tensor(self)
            not_storing_instances = list(set(self.instance_priorities.keys()) - set(storing_instances))
            not_storing_priority = sum(
                (self.instance_priorities[not_storing_instance] for not_storing_instance in not_storing_instances)
            )
            return not_storing_priority

    def initialize_instance_priorities(
        self, G: "ComputationNodeWorkload", node: "ComputationNode", accelerator: "Accelerator"
    ):
        if self.layer_operand == node.output_operand:
            out_edges = [(succ, d) for n, succ, d in G.out_edges(node, data=True) if succ.id != n.id]
            for successor, data in out_edges:
                core = accelerator.get_core(successor.chosen_core_allocation)
                layer_operand = data["operand"]
                memory_operand = successor.memory_operand_links.layer_to_mem_op(layer_operand)
                top_instance = core.get_top_memory_instance(memory_operand)
                if top_instance in self.instance_priorities:
                    self.instance_priorities[top_instance] += 1
                else:  # first time we see this instance
                    self.instance_priorities[top_instance] = 1

        else:
            core = accelerator.get_core(node.chosen_core_allocation)
            memory_operand = self.memory_operand
            top_instance = core.get_top_memory_instance(memory_operand)
            self.instance_priorities[top_instance] = self.base_priority

    def get_total_priority(self):
        return sum(self.instance_priorities.values())

    @property
    def source(self) -> SubviewOp | AllocOp:
        return self.subview.source.op

    @property
    def original_shape(self) -> list[int]:
        """Get the original shape of the tensor before subviewing."""
        source = self.source
        while isinstance(source, SubviewOp):
            source = source.source.op
        assert isinstance(source, AllocOp)
        results_type: MemRefType = source.results[0].type
        return results_type.get_shape()
