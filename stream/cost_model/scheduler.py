import logging
from operator import itemgetter
from typing import TYPE_CHECKING

from zigzag.datatypes import Constants, LayerOperand, MemoryOperand

from stream.hardware.architecture.core import Core
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload
from stream.workload.tensor import SubviewTensor

if TYPE_CHECKING:
    from stream.hardware.architecture.accelerator import Accelerator

logger = logging.getLogger(__name__)


def initialize_priorities(workload: ComputationNodeWorkload, accelerator: "Accelerator"):
    for n in workload.node_list:
        for tensor in n.operand_tensors.values():
            tensor.initialize_instance_priorities(workload, n, accelerator)


def initialize_offchip_tensors(workload: ComputationNodeWorkload, accelerator: "Accelerator"):
    offchip_core_id = accelerator.offchip_core_id
    assert offchip_core_id is not None, "No offchip core found for this accelerator"
    offchip_core = accelerator.get_core(offchip_core_id)
    offchip_top_instances = accelerator.get_top_instances_of_core(offchip_core_id)
    for n in workload.node_list:
        for op, tensor in n.operand_tensors.items():
            # For constant operands or inputs of first node
            if op in n.constant_operands or (op != Constants.OUTPUT_LAYER_OP and len(workload.in_edges(n)) == 0):
                if not any(
                    (
                        accelerator.contains_tensor(tensor, offchip_top_instance)
                        for offchip_top_instance in offchip_top_instances
                    )
                ):
                    memory_op = n.memory_operand_links.layer_to_mem_op(op)
                    accelerator.spawn(
                        tensor=tensor,
                        core=offchip_core,
                        memory_op=memory_op,
                        initial_timestep=0,
                        available_timestep=0,
                    )


def prefetch_constant_operands(
    G: ComputationNodeWorkload, accelerator: "Accelerator", operands_to_prefetch: list[LayerOperand]
):
    total_cn_offchip_link_energy = 0
    total_cn_offchip_memory_energy = 0
    total_eviction_to_offchip_link_energy = 0
    total_eviction_to_offchip_memory_energy = 0
    for n in G.node_list:
        for op, tensor in n.operand_tensors.items():
            if op in n.constant_operands and op in operands_to_prefetch:
                core_allocation = n.chosen_core_allocation
                assert core_allocation is not None, "Core should be allocated"
                memory_op = n.memory_operand_links.layer_to_mem_op(op)
                if not accelerator.contains_tensor(tensor, core_allocation):
                    (
                        _,
                        transfer_link_energy_cost,
                        transfer_memory_energy_cost,
                        eviction_link_energy_cost,
                        eviction_memory_energy_cost,
                        came_from_offchip,
                    ) = accelerator.transfer_tensor_to_core(tensor, core_allocation, memory_op, [])
                    assert came_from_offchip
                    total_cn_offchip_link_energy += transfer_link_energy_cost
                    total_cn_offchip_memory_energy += transfer_memory_energy_cost
                    total_eviction_to_offchip_link_energy += eviction_link_energy_cost
                    total_eviction_to_offchip_memory_energy += eviction_memory_energy_cost
    return (
        total_cn_offchip_link_energy,
        total_cn_offchip_memory_energy,
        total_eviction_to_offchip_link_energy,
        total_eviction_to_offchip_memory_energy,
    )


def get_best_candidate(
    candidates: list[tuple[int, ComputationNode]], scheduling_order: list[tuple[int, int]]
) -> tuple[ComputationNode, int]:
    # If this core doesn't have any candidates, continue to the next core
    if not candidates:
        raise ValueError("There are no candidates to schedule.")
    preds_ends, cn_candidates = zip(*candidates)
    cn_candidates: list[ComputationNode]
    idxs = [scheduling_order.index((n.id, n.sub_id)) for n in cn_candidates]
    best_candidate_idx = idxs.index(min(idxs))
    best_candidate = cn_candidates[best_candidate_idx]
    preds_end = preds_ends[best_candidate_idx]
    # Remove the candidate from the list of candidates
    del candidates[best_candidate_idx]
    return best_candidate, preds_end


def get_tensors_needed_for_node(node: ComputationNode, G: ComputationNodeWorkload):
    """Determine all the tensors needed to compute a node.
    The node might need multiple outputs from previous nodes, depending on the graph.

    Args:
        node (ComputationNode): The node to be computed.
        G : The graph of all nodes.

    Returns:
        tuple: A tuple of tensors and a tuple of memory operands for the node.
    """
    tensors_this_candidate_needs: list[SubviewTensor] = []
    tensors_operands: list[MemoryOperand] = []
    # Constant input operands
    for layer_op in node.constant_operands:
        memory_op = node.memory_operand_links.layer_to_mem_op(layer_op)
        if memory_op in node.too_large_operands:
            continue
        tensors_this_candidate_needs.append(node.operand_tensors[layer_op])
        tensors_operands.append(memory_op)
    # Non-constant input operands
    for pred, node, edge_data in sorted(G.in_edges(node, data=True), key=itemgetter(0)):
        if pred.id == node.id:
            continue  # Skip if predecessor was from the same layer (intra-edge)
        consumer_layer_op = edge_data["operand"]
        consumer_memory_op = node.memory_operand_links[consumer_layer_op]
        if consumer_memory_op in node.too_large_operands:
            continue  # Skip if tensor will be fetched fromm offchip throughout computation
        pred_output_tensor = pred.operand_tensors[pred.output_operand]
        tensors_this_candidate_needs.append(pred_output_tensor)
        tensors_operands.append(consumer_memory_op)
    if tensors_this_candidate_needs:
        tensors_this_candidate_needs, tensors_operands = zip(
            *sorted(zip(tensors_this_candidate_needs, tensors_operands))
        )
    return tensors_this_candidate_needs, tensors_operands


def clear_memories(
    accelerator: "Accelerator",
    core: Core,
    memory_operands: list[MemoryOperand],
    timestep: int,
    exceptions: list[SubviewTensor] = [],
):
    total_eviction_to_offchip_link_energy = 0
    total_eviction_to_offchip_memory_energy = 0
    for too_large_operand in memory_operands:
        (
            timestep,
            eviction_link_energy_cost,
            eviction_memory_energy_cost,
        ) = accelerator.remove_all(core, too_large_operand, timestep, exceptions, write_back_to_offchip=True)
        total_eviction_to_offchip_link_energy += eviction_link_energy_cost
        total_eviction_to_offchip_memory_energy += eviction_memory_energy_cost
    return (
        total_eviction_to_offchip_link_energy,
        total_eviction_to_offchip_memory_energy,
        timestep,
    )


def decrease_priority(
    tensors: list[SubviewTensor],
    tensors_operands: list[MemoryOperand],
    accelerator: "Accelerator",
    node: ComputationNode,
):
    for tensor_used_by_node, tensor_memory_operand in zip(tensors, tensors_operands):
        # TODO: tensor_memory_operand will be 'O' for activation tensors.
        # TODO: If the memory between input and output is not shared, this will give a wrong instance.
        assert node.chosen_core_allocation is not None
        top_instance = accelerator.get_top_instance_of_core(node.chosen_core_allocation, tensor_memory_operand)
        tensor_used_by_node.instance_priorities[top_instance] -= 1


def check_for_removal(
    tensors: list[SubviewTensor],
    accelerator: "Accelerator",
    node: ComputationNode,
    G: ComputationNodeWorkload,
    timestep: int,
):
    offchip_core_id = accelerator.offchip_core_id
    for tensor_used_by_node in tensors:
        if tensor_used_by_node.get_total_priority() == 0:
            instances_storing_tensor, _ = accelerator.memory_manager.find_tensor_in_top_instances(tensor_used_by_node)
            for instance_storing_tensor in instances_storing_tensor:
                core_ids_of_instance = [
                    core.id for core in accelerator.memory_manager.cores_per_top_instance[instance_storing_tensor]
                ]
                # If this tensor is an output tensor, find all nodes that needed it
                # to get an accurate timestep at which it can be removed
                timestep_for_removal = timestep
                if tensor_used_by_node.layer_operand == tensor_used_by_node.cn_source.output_operand:
                    origin = tensor_used_by_node.cn_source
                    if offchip_core_id in core_ids_of_instance:
                        # If wanting to discard it from offchip, look at the max end time across all successors
                        nodes_that_needed_tensor = [n for n in G.successors(origin) if n.id != origin.id]
                    else:
                        # If discarding it from a regular core, look at the max end time successors that used it from
                        # that instance
                        nodes_that_needed_tensor = [
                            n
                            for n in G.successors(origin)
                            if n.chosen_core_allocation in core_ids_of_instance and n.id != origin.id
                        ]
                    end_times = [n.end for n in nodes_that_needed_tensor if n.end is not None]
                    max_end_time = max(end_times, default=timestep_for_removal)
                    # assert max_end_time != -1, "There should be at least one successor."
                    timestep_for_removal = max_end_time

                # Get a core tied to the top_instance we want to remove it on.
                core = accelerator.memory_manager.cores_per_top_instance[instance_storing_tensor][0]
                accelerator.remove(
                    tensor_used_by_node,
                    core,
                    tensor_used_by_node.memory_operand,
                    timestep_for_removal,
                )


def schedule_graph(
    G: ComputationNodeWorkload,
    accelerator: "Accelerator",
    cores_idle_from: dict[int, int] | None = None,
    operands_to_prefetch: list[LayerOperand] = [],
    scheduling_order: list[tuple[int, int]] | None = None,
) -> tuple[int, float, float, float, float, float, float, float, float, float]:
    """Schedule the nodes of graph G across the cores in the system.
    Each node should have a core_allocation and runtime set.

    Args:
        G : Graph containing the nodes to be scheduled.
        accelerator (Accelerator): The accelerator to schedule the nodes on.
        cores_start_offset (dict, optional): A dict containing for each core_id its start offset. Defaults to None.
        operands_to_prefetch (list, optional): The layer operands that should be prefetched at the start of the
            schedule.
    """
    # Initialize total link energy cost and memory energy costs
    total_cn_onchip_energy = 0
    total_cn_offchip_link_energy = 0
    total_cn_offchip_memory_energy = 0
    total_eviction_to_offchip_link_energy = 0
    total_eviction_to_offchip_memory_energy = 0
    total_sink_layer_output_offchip_link_energy = 0
    total_sink_layer_output_offchip_memory_energy = 0
    total_core_to_core_link_energy = 0
    total_core_to_core_memory_energy = 0

    core_ids = set(n.chosen_core_allocation for n in G.node_list)
    assert (
        None not in core_ids
    ), "Make sure all nodes have a core allocation. Insert SetFixedAllocationPerformanceStage."
    all_core_ids: list[int] = sorted(list(core_ids))  # type: ignore

    if cores_idle_from is None:
        # Make it 0 for all cores
        cores_idle_from = {core_allocation: 0 for core_allocation in all_core_ids}

    nb_graph_nodes = G.number_of_nodes()
    nb_scheduled_nodes = 0
    scheduled_nodes: set[ComputationNode] = set()

    # List that keeps all possible candidate nodes for each core.
    candidates: list[tuple[int, ComputationNode]] = []

    # Put the very first nodes of a layer that doesn't have any incoming edges as the first candidates
    for source_node in (n for n, d in G.in_degree() if d == 0):
        core_allocation = source_node.chosen_core_allocation
        candidates.append((cores_idle_from[core_allocation], source_node))  # type: ignore

    # Get all the nodes with no successors that produce final outputs, used for off-loading final outputs
    sink_layers = sorted(set(n.id for n, d in G.out_degree() if d == 0))
    sink_layer_nodes = set((n for n in G.node_list if (n.id in sink_layers) and n.produces_final_output))

    # Get the offchip core id and core
    offchip_core_id = accelerator.offchip_core_id
    assert offchip_core_id is not None
    offchip_core = accelerator.get_core(offchip_core_id)

    # Schedule preparation:
    # 1. Initialize the memory instance priorities for each tensor
    initialize_priorities(G, accelerator)
    # 2. Add the constant operand tensors of all nodes to the off-chip initially
    initialize_offchip_tensors(G, accelerator)
    # 3. Prefetch the constant operands that should be prefetched to their core
    (
        prefetch_cn_offchip_link_energy,
        prefetch_cn_offchip_memory_energy,
        prefetch_eviction_to_offchip_link_energy,
        prefetch_eviction_to_offchip_memory_energy,
    ) = prefetch_constant_operands(G, accelerator, operands_to_prefetch)
    total_cn_offchip_link_energy += prefetch_cn_offchip_link_energy
    total_cn_offchip_memory_energy += prefetch_cn_offchip_memory_energy
    total_eviction_to_offchip_link_energy += prefetch_eviction_to_offchip_link_energy
    total_eviction_to_offchip_memory_energy += prefetch_eviction_to_offchip_memory_energy

    done = False
    while not done:
        # Get the best candidate given the selection priority
        best_candidate, preds_end = get_best_candidate(candidates, scheduling_order)

        # Get the core this candidate will be scheduled on
        core_id = best_candidate.chosen_core_allocation
        assert core_id is not None
        core = accelerator.get_core(core_id)
        # Earliest start time is when core is available or predecessors finished
        start = max(cores_idle_from[core_id], preds_end)
        # Step 0
        tensors_this_candidate_needs, tensors_operands = get_tensors_needed_for_node(best_candidate, G)
        output_tensor = best_candidate.operand_tensors[best_candidate.output_operand]
        # Step 1
        # There could be operands that are too large to store in the highest memory on the core
        # The tensors stored in these memories should be evicted and potentially written back to off-chip
        # Clear these memories (this might delay the potential start time if things have to written to off-chip)
        timestep = start
        (
            clear_link_energy,
            clear_memory_energy,
            timestep,
        ) = clear_memories(
            accelerator,
            core,
            best_candidate.too_large_operands,
            timestep,
            exceptions=tensors_this_candidate_needs + (output_tensor,),
        )
        total_eviction_to_offchip_link_energy += clear_link_energy
        total_eviction_to_offchip_memory_energy += clear_memory_energy
        # Step 2
        # The computation might need tensors that are currently not present in the core's memories
        # We need to fetch these tensors from either off-chip or from the core where they are present
        # Transfer these tensors from wherever they are currently residing to this core
        print(f"Scheduling {best_candidate} on core {core_id} at timestep {timestep}")
        for tensor, tensor_operand in zip(tensors_this_candidate_needs, tensors_operands):
            # Transfer the tensor
            (
                transfer_complete_timestep,
                transfer_link_energy_cost,
                transfer_memory_energy_cost,
                eviction_link_energy_cost,
                eviction_memory_energy_cost,
                came_from_offchip,
            ) = accelerator.transfer_tensor_to_core(
                tensor,
                core_id,
                tensor_operand,
                tensors_this_candidate_needs + (output_tensor,),
            )
            # Update the possible start time of this node
            timestep = max(timestep, transfer_complete_timestep)
            # Add the energy costs to their respective trackers
            if came_from_offchip:
                total_cn_offchip_link_energy += transfer_link_energy_cost
                total_cn_offchip_memory_energy += transfer_memory_energy_cost
            else:
                total_core_to_core_link_energy += transfer_link_energy_cost
                total_core_to_core_memory_energy += transfer_memory_energy_cost
            total_eviction_to_offchip_link_energy += eviction_link_energy_cost
            total_eviction_to_offchip_memory_energy += eviction_memory_energy_cost

        # Step 3
        # Check if we had any operands that were too large to store in the core's memory, block the relevant off-chip
        # link for the duration
        # This might again delay the execution if the offchip link was already blocked by another core
        timestep = accelerator.block_offchip_links(
            best_candidate.too_large_operands,
            core_id,
            timestep,
            best_candidate.get_runtime(),
            best_candidate,
        )

        # Step 4
        # Make space for the output tensor of this computation node and spawn it when evictions are complete
        # If the output operand is in the too large operands, add it to off-chip, otherwise add it to this core's
        # output memory
        output_layer_operand = best_candidate.output_operand
        output_memory_operand = best_candidate.memory_operand_links[output_layer_operand]
        if output_memory_operand in best_candidate.too_large_operands:
            core_to_add_output_to = offchip_core
        else:
            core_to_add_output_to = core
        (
            evictions_complete_timestep,
            eviction_link_energy_cost,
            eviction_memory_energy_cost,
        ) = accelerator.make_space_for(
            output_tensor,
            core_to_add_output_to,
            output_memory_operand,
            timestep,
            tensors_this_candidate_needs + (output_tensor,),
        )
        total_eviction_to_offchip_link_energy += eviction_link_energy_cost
        total_eviction_to_offchip_memory_energy += eviction_memory_energy_cost
        start = evictions_complete_timestep
        end = start + best_candidate.get_runtime()
        accelerator.spawn(
            output_tensor,
            core_to_add_output_to,
            output_memory_operand,
            initial_timestep=start,
            available_timestep=end,
        )
        # Fix the memory usage and nb of stored tensors until the CN ends
        accelerator.fix_memory_usage_and_nb_stored_tensors(
            core,
            start,
            end,
        )

        # Step 5
        # Update the start and end time of the node
        best_candidate.set_start(start)
        best_candidate.set_end(end)
        cores_idle_from[core_id] = end

        # Add the computation energy of running this node
        total_cn_onchip_energy += best_candidate.get_onchip_energy()
        total_cn_offchip_memory_energy += best_candidate.get_offchip_energy()

        # Add this node to the scheduled nodes
        scheduled_nodes.add(best_candidate)

        # Step 6
        # Memory usage: When the node ends:
        # Decrease the priority of all the tensors this node used
        decrease_priority(tensors_this_candidate_needs, tensors_operands, accelerator, best_candidate)
        # Remove the tensor if the priority is zero
        check_for_removal(
            tensors_this_candidate_needs,
            accelerator,
            best_candidate,
            G,
            end,
        )

        # Step 7
        # Memory usage: When the node ends:
        # If this node is a sink node (node that has no successors and that produces a final output), transfer final
        # outputs to offchip
        if best_candidate in sink_layer_nodes:
            # Only push back sink node outputs if they're generated and stored on the core
            if best_candidate.output_operand not in best_candidate.too_large_operands:
                (
                    _,
                    link_energy_cost,
                    memory_energy_cost,
                ) = accelerator.remove(
                    output_tensor,
                    core,
                    output_tensor.memory_operand,
                    end,
                    write_back_to_offchip=True,
                )
                total_sink_layer_output_offchip_link_energy += link_energy_cost
                total_sink_layer_output_offchip_memory_energy += memory_energy_cost

        # Step 8
        # For each successor of this node, check if all of its predecessors have been scheduled
        for successor in sorted(G.successors(best_candidate)):
            if all((pred in scheduled_nodes for pred in G.predecessors(successor))):
                preds_end = max(
                    (predecessor.end for predecessor in G.predecessors(successor)),
                    default=0,
                )
                # core_candidates[successor.core_allocation].append((preds_end, successor))
                candidates.append((preds_end, successor))

        # Increment the number of scheduled nodes
        nb_scheduled_nodes += 1
        done = nb_scheduled_nodes == nb_graph_nodes

    # Step 9
    # The total schedule latency is the max of all CN end times and the link end times
    cns_end_time = max((n.end for n in G.node_list))
    links_end_time = max([event.end for event in accelerator.communication_manager.events], default=0)
    latency = max(cns_end_time, links_end_time)

    return (
        latency,
        total_cn_onchip_energy,
        total_cn_offchip_link_energy,
        total_cn_offchip_memory_energy,
        total_eviction_to_offchip_link_energy,
        total_eviction_to_offchip_memory_energy,
        total_sink_layer_output_offchip_link_energy,
        total_sink_layer_output_offchip_memory_energy,
        total_core_to_core_link_energy,
        total_core_to_core_memory_energy,
    )
