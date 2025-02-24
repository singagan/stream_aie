from zigzag.datatypes import Constants
from zigzag.workload.layer_node_abc import LayerNodeABC

from stream.node_tensor import NodeTensor
from stream.workload.node import Node


class ReshapeNode(Node, LayerNodeABC):
    """Class that represents an onnx Reshape node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessor: int,
        shape: tuple[int, ...],
        allow_zero: bool = False,
    ) -> None:
        """Initialize the ReshapeNode

        Args:
            predecessors: The id of this node's parent.
            shape: The output tensor's shape.
            allow_zero: wether the output shape can be 0 at some dimensions. Iff True, shape `[2,0,3]` becomes `[2,3]`
        """
        Node.__init__(
            self,
            node_id=node_id,
            node_name=node_name,
            type="reshape",
            onchip_energy=0,
            offchip_energy=0,
            runtime=0,
            possible_core_allocation=[-1],
        )
        LayerNodeABC.__init__(self, node_id=node_id, node_name=node_name)

        self.allow_zero = allow_zero
        self.shape = shape
        self.input_operand_source = {Constants.LAYER_OP_I: predecessor}

    def reshape_operand_tensor(self, tensor: NodeTensor):
        """Reshape the tensor back to the representation needed for producer/consumer."""
        new_shape = self.shape
        if not new_shape:
            return tensor

        if not self.allow_zero:
            new_shape = tuple(x for x in new_shape if x != 0)
        return tensor.reshape(new_shape)
