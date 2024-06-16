import numpy as np
import networkx as nx
from networkx import DiGraph

#from stream.classes.hardware.architecture.communication_link import CommunicationLink
from stream.classes.hardware.architecture.noc import CommunicationLink

# import from stream_core class instead
# from zigzag.classes.hardware.architecture.core import Core
from stream.classes.hardware.architecture.stream_core import Core

# From the AIE-MLs perspective, the throughput of each of the loads and store is 256 bits per clock cycle.
aya_core_to_core_bw = 256  # bandwidth of every link connecting two neighboring cores

def have_shared_memory(a, b):
    """Returns True if core a and core b have a shared top level memory

    Args:
        a (Core): First core
        b (Core): Second core
    """
    top_level_memory_instances_a = set(
        [
            level.memory_instance
            for level, out_degree in a.memory_hierarchy.out_degree()
            if out_degree == 0
        ]
    )
    top_level_memory_instances_b = set(
        [
            level.memory_instance
            for level, out_degree in b.memory_hierarchy.out_degree()
            if out_degree == 0
        ]
    )
    for memory_instance_a in top_level_memory_instances_a:
        if memory_instance_a in top_level_memory_instances_b:
            return True
    return False


def get_2d_mesh(
    cores: list[Core],
    nb_rows: int,
    nb_cols: int,
    bandwidth: int,
    unit_energy_cost: float,
    pooling_core: Core | None = None,
    simd_core: Core | None = None,
    offchip_core: Core | None = None,
):
    """Return a 2D mesh graph of the cores where each core is connected to its N, E, S, W neighbour.
    We build the mesh by iterating through the row and then moving to the next column.
    Each connection between two cores includes two links, one in each direction, each with specified bandwidth.
    Thus there are a total of ((nb_cols-1)*2*nb_rows + (nb_rows-1)*2*nb_cols) links in the noc.
    If a pooling_core is provided, it is added with two directional links with each core, one in each direction.
    Thus, 2*nb_rows*nb_cols more links are added.
    If an offchip_core is provided, it is added with two directional links with each core, one in each direction.
    Thus, 2*nb_rows*nb_cols (+2 if a pooling core is present)

    Args:
        cores (list): list of core objects
        nb_rows (int): the number of rows in the 2D mesh
        nb_cols (int): the number of columns in the 2D mesh
        bandwidth (int): bandwidth of each created directional link in bits per clock cycle
        unit_energy_cost (float): The unit energy cost of having a communication-link active. This does not include the involved memory read/writes.
        pooling_core (Core, optional): If provided, the pooling core that is added.
        simd_core (Core, optional): If provided, the simd core that is added.
        offchip_core (Core, optional): If provided, the offchip core that is added.
        offchip_bandwidth (int, optional): If offchip_core is provided, this is the
    """
    ########### Beginning of the logic for adding the links representing the shared memory
    # At the moment there is a shared memory link in 4 directions

    use_shared_mem_flag = True
    
    cores_array = np.asarray(cores).reshape((nb_rows, nb_cols), order="C")
    edges = []
    # Horizontal edges
    for row in cores_array:
        # From left to right
        pairs = zip(row, row[1:])
        for pair in pairs:
            (sender, receiver) = pair
            # Aya
            if(sender.core_type == 1 or receiver.core_type == 1):  # skip memTile cores
                continue
            if use_shared_mem_flag:
                if not have_shared_memory(sender, receiver):
                    edges.append(
                        (
                            sender,
                            receiver,
                            {
                                "cl": CommunicationLink(
                                    sender, receiver, aya_core_to_core_bw, unit_energy_cost
                                )
                            },
                        )
                    )

        # From right to left
        pairs = zip(reversed(row), reversed(row[:-1]))
        for pair in pairs:
            (sender, receiver) = pair
            # Aya
            if(sender.core_type == 1 or receiver.core_type == 1):  # skip memTile cores
                continue
            if use_shared_mem_flag:
                if not have_shared_memory(sender, receiver):
                    edges.append(
                        (
                            sender,
                            receiver,
                            {
                                "cl": CommunicationLink(
                                    sender, receiver, aya_core_to_core_bw, unit_energy_cost
                                )
                            },
                        )
                    )
           
    # Vertical edges
    for col in cores_array.T:
        # From top to bottom (bottom is highest idx)
        pairs = zip(col, col[1:])
        for pair in pairs:
            (sender, receiver) = pair
            # Aya
            if(sender.core_type == 1 or receiver.core_type == 1):  # skip memTile cores
                continue
           
            if use_shared_mem_flag:
                if not have_shared_memory(sender, receiver):
                    edges.append(
                        (
                            sender,
                            receiver,
                            {
                                "cl": CommunicationLink(
                                    sender, receiver, aya_core_to_core_bw, unit_energy_cost
                                )
                            },
                        )
                    )
            
        # From bottom to top
        pairs = zip(reversed(col), reversed(col[:-1]))
        for pair in pairs:
            # Aya
            (sender, receiver) = pair
            if(sender.core_type == 1 or receiver.core_type == 1):  # skip memTile cores
                continue
            if use_shared_mem_flag:
                if not have_shared_memory(sender, receiver):
                    edges.append(
                        (
                            sender,
                            receiver,
                            {
                                "cl": CommunicationLink(
                                    sender, receiver, aya_core_to_core_bw, unit_energy_cost
                                )
                            },
                        )
                    )
    ########### End of the logic for adding the links representing the shared memory
                
    # If there is an offchip core, add a single link for writing to and a single link for reading from the offchip
    if offchip_core:
        offchip_read_bandwidth = offchip_core.mem_r_bw_dict["O"][0]
        offchip_write_bandwidth = offchip_core.mem_w_bw_dict["O"][0]
        # if the offchip core has only one port
        if len(offchip_core.mem_hierarchy_dict["O"][0].port_list) == 1:
            to_offchip_link = CommunicationLink(
                offchip_core,
                "Any",
                offchip_write_bandwidth,
                unit_energy_cost,
                bidirectional=True,
            )
            from_offchip_link = to_offchip_link
        # if the offchip core has more than one port
        else:
            to_offchip_link = CommunicationLink(
                "Any", offchip_core, offchip_write_bandwidth, unit_energy_cost
            )
            from_offchip_link = CommunicationLink(
                offchip_core, "Any", offchip_read_bandwidth, unit_energy_cost
            )
        if not isinstance(offchip_core, Core):
            raise ValueError("The given offchip_core is not a Core object.")
        for core in cores:
            edges.append((core, offchip_core, {"cl": to_offchip_link}))
            edges.append((offchip_core, core, {"cl": from_offchip_link}))
        if pooling_core:
            edges.append((pooling_core, offchip_core, {"cl": to_offchip_link}))
            edges.append((offchip_core, pooling_core, {"cl": from_offchip_link}))
        if simd_core:
            edges.append((simd_core, offchip_core, {"cl": to_offchip_link}))
            edges.append((offchip_core, simd_core, {"cl": from_offchip_link}))

    # Build the graph using the constructed list of edges
    H = DiGraph(edges)

    return H
