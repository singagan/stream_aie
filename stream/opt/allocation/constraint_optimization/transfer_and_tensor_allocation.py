# transfer_and_tensor_allocator.py
import math
from collections import defaultdict
from math import ceil
from typing import Any

import gurobipy as gp
from gurobipy import GRB, quicksum

from stream.hardware.architecture.core import Core
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.opt.allocation.constraint_optimization.timeslot_allocation import (
    Resource,
    TimeSlotAllocation,
    _resource_key,
)
from stream.workload.steady_state.computation import SteadyStateComputation
from stream.workload.steady_state.iteration_space import IterationVariableReuse
from stream.workload.steady_state.node import SteadyStateNode
from stream.workload.steady_state.tensor import SteadyStateTensor
from stream.workload.steady_state.transfer import SteadyStateTransfer
from stream.workload.steady_state.workload import SteadyStateWorkload


class TransferAndTensorAllocator:
    """
    MILP that decides

    1. **where** every *movable* SteadyStateTensor lives,
    2. **which path** each SteadyStateTransfer uses,

    while all slots/resources coming from an existing TimeSlotAllocation
    remain unchanged.
    """

    # ------------------------------------------------------------ #
    # ctor / public API                                            #
    # ------------------------------------------------------------ #
    def __init__(
        self,
        ssw: SteadyStateWorkload,
        tsa: TimeSlotAllocation,
        *,
        offchip_core_id: int | None = None,
        iterations: int = 1,
        big_m: int | None = None,
        gurobi_verbosity: int = 1,
    ):
        self.ssw = ssw
        self.tsa = tsa
        self.offchip_core_id = offchip_core_id
        self.iterations = iterations
        self.max_slot = tsa.slot_max
        self.big_m = big_m or len(ssw.nodes()) + 5

        # ------------------- categorise nodes -------------------- #
        self.ssc_nodes: list[SteadyStateComputation] = ssw.computation_nodes
        self.transfer_nodes: list[SteadyStateTransfer] = ssw.transfer_nodes

        # Fixed-vs-movable tensors
        self.tensor_fixed: list[SteadyStateTensor] = []
        self.tensor_var: list[SteadyStateTensor] = []
        for t in ssw.tensor_nodes:
            if len(t.possible_resource_allocation or []) <= 1 or t.chosen_resource_allocation is not None:
                self.tensor_fixed.append(t)
            else:
                self.tensor_var.append(t)
        assert not self.tensor_var, "Variable tensors are not supported since transfer firing changes."

        # quick look-ups from TSA
        self.slot_of: dict[Any, int] = {n: tsa.get_timeslot_of_node(n) for n in tsa.nodes}
        self.resource_of: dict[Any, Resource] = {n: next(iter(tsa.get_resources_for_node(n))) for n in tsa.nodes}

        # --------------- optimisation model ---------------------- #
        self.model = gp.Model("transfer_tensor_alloc")
        self.model.setParam("OutputFlag", gurobi_verbosity)

        # decision vars
        self.x_tensor: dict[tuple[SteadyStateTensor, Core], gp.Var] = {}
        self.y_path: dict[tuple[SteadyStateTransfer, tuple[CommunicationLink, ...]], gp.Var] = {}

        # helpers
        self.link_set: set[CommunicationLink] = set()
        self.links_in_path: dict[
            tuple[SteadyStateTransfer, tuple[CommunicationLink, ...]],
            list[CommunicationLink],
        ] = {}

        # latency vars
        self.slot_latency: dict[int, gp.Var] = {}
        self.overlap = None
        self.total_latency = None

        # transfer fire helpers init
        # dict((transfer, steady state iteration space index): (fires_across_ss_iterations, extra_mem_bytes))
        self._ensure_same_ssis_for_all_transfers()
        self.reuse_levels: dict[tuple[SteadyStateTransfer, int], tuple[int, int]] = {}
        self.transfer_nodes_to_optimize_firings_for: list[SteadyStateTransfer] = []
        self._init_transfer_fire_helpers()

        self._build_model()

    # ------------------------------------------------------------ #
    # internal helpers                                             #
    # ------------------------------------------------------------ #
    # ---------- memory factor ( NO ping-pong for now ) ---------- #
    # TODO: Update this for memory-only cores
    @staticmethod
    def _mem_factor(t: SteadyStateTensor, core: Core) -> int:
        # if getattr(core, "type", None) == "COMPUTE":
        return 1  # NO ping–pong
        # else:
        #     full_fac = getattr(t, "slices_per_full", 1)
        #     return int(full_fac)

    # ---------- latency of a transfer along a path -------------- #
    @staticmethod
    def _transfer_latency(tr: SteadyStateTransfer, path: tuple[CommunicationLink, ...]) -> int:
        if not path:
            return 0
        min_bw = min(link.bandwidth for link in path)
        return ceil(tr.size / min_bw)

    # ---------- init transfer fire helpers ----------------------- #
    def _ensure_same_ssis_for_all_transfers(self) -> None:
        """
        Ensure that all transfers have the same steady-state iteration space dimensions and sizes (SSIS).
        """
        ssis_dims_sizes = [
            (iter_var.dimension, iter_var.size) for iter_var in self.transfer_nodes[0].steady_state_iteration_space
        ]
        for tr in self.transfer_nodes:
            dims_sizes = [(iter_var.dimension, iter_var.size) for iter_var in tr.steady_state_iteration_space]
            if dims_sizes != ssis_dims_sizes:
                raise ValueError(
                    f"Transfer {tr.node_name} has a different SSIS than the first transfer {self.transfer_nodes[0]}: "
                    f"{ssis_dims_sizes} != {ssis_dims_sizes}"
                )

    def _init_transfer_fire_helpers(self) -> None:
        for tr in self.transfer_nodes:  # only the movable tensors
            ssis = tr.steady_state_iteration_space  # e.g. [Nk, Nk-1, …, N0]
            sizes = [iter_var.size for iter_var in ssis]
            relevancies = [iter_var.relevant for iter_var in ssis]
            reuses = [iter_var.reuse for iter_var in ssis]

            # Check that all reuses are NOT_SET, else continue
            if any(r != IterationVariableReuse.NOT_SET for r in reuses):
                continue
            self.transfer_nodes_to_optimize_firings_for.append(tr)

            # level = -1  →  "transfer every steady state iteration"
            fires = math.prod(sizes)
            mem_needed = tr.size
            self.reuse_levels[(tr, -1)] = (fires, mem_needed)
            # level = i  →  keep while loops 0..i stay in cache
            for i, (Nl, relevancy) in enumerate(zip(sizes, relevancies, strict=True)):  # i = 0 … K-1
                mem_needed *= Nl if relevancy else 1  # enlarge tile size factor only if relevant
                fires //= Nl  # fewer transfers
                self.reuse_levels[(tr, i)] = (fires, mem_needed)

    # ------------------------------------------------------------ #
    # model construction                                           #
    # ------------------------------------------------------------ #
    def _build_model(self):
        self._create_vars()
        self._path_choice_constraints()
        self._tensor_placement_constraints()
        self._transfer_fire_rate_constraints()
        self._link_contention_constraints()
        self._memory_capacity_constraints()
        self._slot_latency_constraints()
        self._overlap_and_objective()

    # ...................... variable creation ................... #
    def _create_vars(self):
        # ---- tensor placement ---------------------------------- #
        for t in self.tensor_var:
            for c in t.possible_resource_allocation:  # type: ignore
                v = self.model.addVar(vtype=GRB.BINARY, name=f"x_{t.node_name}_{_resource_key(c)}")
                self.x_tensor[(t, c)] = v

        # ---- transfer path choice ------------------------------ #
        for tr in self.transfer_nodes:
            for p in tr.possible_resource_allocation:  # list[list[Link]]
                p_tuple = tuple(p)
                v = self.model.addVar(vtype=GRB.BINARY, name=f"y_{tr.node_name}_{hash(p_tuple)}")
                self.y_path[(tr, p_tuple)] = v
                self.links_in_path[(tr, p_tuple)] = list(p_tuple)
                self.link_set.update(p_tuple)

        # ---- slot latency vars --------------------------------- #
        for s in range(self.max_slot + 1):
            self.slot_latency[s] = self.model.addVar(vtype=GRB.INTEGER, name=f"L_{s}")

        # -- outer loop to cache tensors and skip transfers for -- #
        self.z_cache: dict[tuple[SteadyStateTransfer, int], gp.Var] = {}
        for tr in self.transfer_nodes:
            sizes = [iter_var.size for iter_var in tr.steady_state_iteration_space]
            for stop in range(-1, len(sizes)):  # -1 .. K-1
                v = self.model.addVar(vtype=GRB.BINARY, name=f"zStop_{tr.node_name}_L{stop}")
                self.z_cache[(tr, stop)] = v
            # Choose exactly one stop-level
            self.model.addConstr(
                quicksum(self.z_cache[(tr, s)] for s in range(-1, len(sizes))) == 1,
                name=f"cacheChooseStop_{tr.node_name}",
            )
            if tr not in self.transfer_nodes_to_optimize_firings_for:
                # Get the stop value by looking at the reuses of the ssis
                reuses = [iter_var.reuse for iter_var in tr.steady_state_iteration_space]
                # Find the index of the last 'REUSE' in the reuses
                stop = -2
                for i in range(len(reuses) - 1, -1, -1):
                    if reuses[i] == IterationVariableReuse.REUSE:
                        stop = i
                        break
                assert stop >= -1, f"Something went wrong for Transfer {tr.node_name} REUSE indexing: {reuses}"
                # Set the z_cache variable to 1 for the chosen stop
                self.model.addConstr(
                    self.z_cache[(tr, stop)] == 1,
                    name=f"cacheFixedStop_{tr.node_name}_L{stop}",
                )

    # ...................... tensor placement .................... #
    def _tensor_placement_constraints(self):
        for t in self.tensor_var:
            self.model.addConstr(
                quicksum(self.x_tensor[(t, c)] for c in t.possible_resource_allocation) == 1,  # type: ignore
                name=f"place_{t.node_name}",
            )

    def _transfer_fire_rate_constraints(self):
        self.fires: dict[SteadyStateTransfer, gp.Var] = {}
        for tr in self.transfer_nodes:
            fires_tr = self.model.addVar(vtype=GRB.INTEGER, name=f"fires_{tr.node_name}")
            self.fires[tr] = fires_tr
            self.model.addConstr(
                fires_tr
                == quicksum(
                    self.reuse_levels[(tr, stop)][0] * self.z_cache[(tr, stop)]
                    for stop in range(-1, len(tr.steady_state_iteration_space))
                ),
                name=f"fires_def_{tr.node_name}",
            )

    # ...................... path choice ........................ #
    def _path_choice_constraints(self) -> None:
        """
        For every transfer, exactly one path must be selected and path choices must be coherent
        with tensor placements for both source and destination tensors.
        """
        for tr in self.transfer_nodes:
            paths = [p for p in self.y_path if p[0] is tr]
            self._add_one_path_constraint(tr, paths)
            self._add_source_tensor_coherence_constraints(tr, paths)
            self._add_destination_tensor_coherence_constraints(tr, paths)

    def _add_one_path_constraint(
        self, tr: SteadyStateTransfer, paths: list[tuple[SteadyStateTransfer, tuple[CommunicationLink, ...]]]
    ) -> None:
        """Ensure exactly one path is selected for each transfer."""
        self.model.addConstr(quicksum(self.y_path[p] for p in paths) == 1, name=f"one_path_{tr.node_name}")

    def _add_source_tensor_coherence_constraints(
        self, tr: SteadyStateTransfer, paths: list[tuple[SteadyStateTransfer, tuple[CommunicationLink, ...]]]
    ) -> None:
        """Add constraints to ensure path choice is coherent with source tensor placement."""
        predecessors = list(self.ssw.predecessors(tr))
        assert all(isinstance(n, SteadyStateTensor) for n in predecessors), (
            f"Transfer {tr.node_name} has non-tensor predecessor(s): {predecessors}"
        )
        successors = list(self.ssw.successors(tr))
        for src_tensor in predecessors:
            if src_tensor in self.tensor_var:
                assert isinstance(src_tensor, SteadyStateTensor), (
                    f"Expected {src_tensor.node_name} to be a SteadyStateTensor, got {type(src_tensor)}"
                )
                for p in paths:
                    if p[1]:
                        src_core = p[1][0].sender  # first link’s source core
                        assert isinstance(src_core, Core), f"Expected {src_core} to be a Core, got {type(src_core)}"
                        self.model.addConstr(
                            self.y_path[p] <= self.x_tensor[(src_tensor, src_core)],
                            name=f"path_core_link_src_{tr.node_name}_{_resource_key(src_core)}",
                        )
                    else:
                        # empty path is only possible if the src_tensor is fixed on the same core as the dst_tensor
                        dst_tensor = successors[0] if successors else None
                        if dst_tensor is not None and dst_tensor in self.tensor_fixed:
                            dst_core = self.resource_of[dst_tensor]
                            assert isinstance(dst_core, Core), f"Expected {dst_core} to be a Core, got {type(dst_core)}"
                            self.model.addConstr(
                                self.y_path[p] <= self.x_tensor[(src_tensor, dst_core)],
                                name=f"path_core_link_src_empty_path_{tr.node_name}_{_resource_key(dst_core)}",
                            )

    def _add_destination_tensor_coherence_constraints(
        self, tr: SteadyStateTransfer, paths: list[tuple[SteadyStateTransfer, tuple[CommunicationLink, ...]]]
    ) -> None:
        """Add constraints to ensure path choice is coherent with destination tensor placement."""
        successors = list(self.ssw.successors(tr))
        assert all(isinstance(n, SteadyStateTensor) for n in successors), (
            f"Transfer {tr.node_name} has non-tensor successor(s): {successors}"
        )
        predecessors = list(self.ssw.predecessors(tr))
        for dst_tensor in successors:
            if dst_tensor in self.tensor_var:
                assert isinstance(dst_tensor, SteadyStateTensor), (
                    f"Expected {dst_tensor.node_name} to be a SteadyStateTensor, got {type(dst_tensor)}"
                )
                for p in paths:
                    if p[1]:
                        dst_core = p[1][-1].receiver  # last link’s destination core
                        if isinstance(dst_core, str):
                            continue  # dst_core is any core, no constraint needed
                        self.model.addConstr(
                            self.y_path[p] <= self.x_tensor[(dst_tensor, dst_core)],
                            name=f"path_core_link_dst_{tr.node_name}_{_resource_key(dst_core)}",
                        )
                    else:
                        # empty path is only possible if the dst_tensor is fixed on the same core as the src_tensor
                        src_tensor = predecessors[0] if predecessors else None
                        if src_tensor is not None and src_tensor in self.tensor_fixed:
                            src_core = self.resource_of[src_tensor]
                            assert isinstance(src_core, Core), f"Expected {src_core} to be a Core, got {type(src_core)}"
                            self.model.addConstr(
                                self.y_path[p] <= self.x_tensor[(dst_tensor, src_core)],
                                name=f"path_core_link_dst_empty_path_{tr.node_name}_{_resource_key(src_core)}",
                            )

    # ...................... link contention .................... #
    def _link_contention_constraints(self):
        # For every link/slot sum of all selected paths ≤ 1
        usage: dict[tuple[CommunicationLink, int], list[gp.Var]] = defaultdict(list)
        for (tr, path_t), y in self.y_path.items():
            s = self.slot_of[tr]
            for link in self.links_in_path[(tr, path_t)]:
                usage[(link, s)].append(y)

        for (link, s), vars_ in usage.items():
            self.model.addConstr(quicksum(vars_) <= 1, name=f"link_usage_{_resource_key(link)}_{s}")

    # ...................... memory capacity .................... #
    def _memory_capacity_constraints(self):
        # start with fixed tensors
        self.core_load: dict[Core, gp.LinExpr] = defaultdict(gp.LinExpr)
        for t in self.tensor_fixed:
            core = self.resource_of[t]
            if isinstance(core, Core):
                self.core_load[core] += self._mem_factor(t, core) * t.size  # type: ignore

        # add variable tensors
        for t in self.tensor_var:
            for c in t.possible_resource_allocation:  # type: ignore
                self.core_load[c] += self._mem_factor(t, c) * t.size * self.x_tensor[(t, c)]  # type: ignore

        # add transfer tensors
        for tr in self.transfer_nodes:
            for t in self.ssw.successors(tr):
                assert isinstance(t, SteadyStateTensor), (
                    f"Expected {t.node_name} to be a SteadyStateTensor, got {type(t)}"
                )
                assert t in self.tensor_fixed, "Post transfer tensors must be fixed (for now)."
                c = self.resource_of[t]
                assert isinstance(c, Core), f"Expected {c} to be a Core, got {type(c)}"
                for stop in range(-1, len(tr.steady_state_iteration_space)):
                    _, size_factor = self.reuse_levels[(tr, stop)]
                    self.core_load[c] += size_factor * self.z_cache[(tr, stop)]

        for c, expr in self.core_load.items():
            cap = c.get_memory_capacity()  # user-provided helper
            self.model.addConstr(expr <= cap, name=f"mem_cap_{_resource_key(c)}")

    # ...................... slot latency ........................ #
    def _slot_latency_constraints(self):
        # constant SSC contributions
        for n in self.ssc_nodes:
            s = self.slot_of[n]
            assert n.runtime is not None, f"Node {n.node_name} has no runtime defined."
            self.model.addConstr(self.slot_latency[s] >= n.runtime, name=f"ssc_lat_{n.node_name}")

        # transfer (path-dependent) contributions
        for (tr, p), y in self.y_path.items():
            s = self.slot_of[tr]
            lat = self._transfer_latency(tr, p)
            self.model.addConstr(self.slot_latency[s] >= lat * y, name=f"tr_lat_{tr.node_name}_{hash(p)}")

    # ...................... overlap + objective ................. #
    def _overlap_and_objective(self) -> None:  # noqa: PLR0915
        """
        ▸ idle_start[res,s]  == 1  ⇔  slot *s* is *before* the first activity on *res*
        ▸ idle_end  [res,s]  == 1  ⇔  slot *s* is *after*  the last  activity on *res*
        Only these slots contribute to the idle-latency that can overlap
        successive steady-state iterations.
        """
        # ------------------------------------------------------------------
        # helpers
        # ------------------------------------------------------------------
        max_s = self.max_slot
        big_m = self.big_m

        def _first_last_busy(res: Resource) -> tuple[int | None, int | None]:
            """Return first / last busy slot of a *fixed* resource."""
            busy = sorted(self.slot_of[n] for n, r in self.resource_of.items() if r is res)
            return (busy[0], busy[-1]) if busy else (None, None)

        # ------------------------------------------------------------------
        # 1) idle-indicator   idleS / idleE
        # ------------------------------------------------------------------
        self.idleS: dict[tuple[Resource, int], gp.Var | int] = {}
        self.idleE: dict[tuple[Resource, int], gp.Var | int] = {}

        # ............................................ fixed cores .........
        for res in self.tsa.resources:
            if res is None or (isinstance(res, Core) and res.id == self.offchip_core_id):
                continue  # not part of overlap

            first_busy, last_busy = _first_last_busy(res)
            if first_busy is None or last_busy is None:
                continue

            for s in range(max_s + 1):
                self.idleS[(res, s)] = 1 if s < first_busy else 0
                self.idleE[(res, s)] = 1 if s > last_busy else 0

        # ............................................ links (path-dep) ....
        # we need:
        #   link_used      = 1  ⇔  link carries traffic in *any* slot
        #   prefix_sum[s]  = Σ_{τ≤s}  active_{τ}
        #   suffix_sum[s]  = Σ_{τ≥s}  active_{τ}
        # idleS = 1  ⇔  prefix_sum[s]==0        (no activity up to *s*)
        # idleE = 1  ⇔  suffix_sum[s]==0        (no activity after *s*)

        self.link_used: dict[CommunicationLink, gp.Var] = {}
        self.prefixs: dict[CommunicationLink, list[gp.Var]] = {}
        self.suffixs: dict[CommunicationLink, list[gp.Var]] = {}

        for link in self.link_set:
            # ---------- active_{link,s} expression ------------------------
            active_s: dict[int, gp.LinExpr] = {}
            for s in range(max_s + 1):
                active_s[s] = quicksum(
                    self.y_path[(tr, p)]
                    for (tr, p) in self.y_path
                    if link in self.links_in_path[(tr, p)] and self.slot_of[tr] == s
                )

            # ---------- link_used binary ----------------------------------
            lu = self.model.addVar(vtype=GRB.BINARY, name=f"linkUsed_{_resource_key(link)}")
            self.link_used[link] = lu
            sum_active = quicksum(active_s.values())
            # (1) if lu == 1  ⇒  link carries traffic
            self.model.addConstr(sum_active >= lu, name=f"link_used_def_{_resource_key(link)}")
            # (2) if link carries traffic  ⇒  lu == 1
            self.model.addConstr(sum_active <= big_m * lu, name=f"link_used_def2_{_resource_key(link)}")

            # ---------- cumulative sums -----------------------------------
            prefix = [
                self.model.addVar(vtype=GRB.INTEGER, name=f"pre_{_resource_key(link)}_{s}") for s in range(max_s + 1)
            ]
            suffix = [
                self.model.addVar(vtype=GRB.INTEGER, name=f"suf_{_resource_key(link)}_{s}") for s in range(max_s + 1)
            ]
            self.prefixs[link] = prefix
            self.suffixs[link] = suffix

            self.model.addConstr(prefix[0] == active_s[0])
            self.model.addConstr(suffix[-1] == active_s[max_s])
            for s in range(1, max_s + 1):
                self.model.addConstr(prefix[s] == prefix[s - 1] + active_s[s])
                self.model.addConstr(suffix[max_s - s] == suffix[max_s - s + 1] + active_s[max_s - s])

            # ---------- idleS / idleE binaries ----------------------------
            for s in range(max_s + 1):
                is_ = self.model.addVar(vtype=GRB.BINARY, name=f"idleS_{_resource_key(link)}_{s}")
                ie_ = self.model.addVar(vtype=GRB.BINARY, name=f"idleE_{_resource_key(link)}_{s}")
                self.idleS[(link, s)] = is_
                self.idleE[(link, s)] = ie_

                # prefix_sum == 0  →  is_ = 1
                self.model.addConstr(prefix[s] <= big_m * (1 - is_))
                self.model.addConstr(prefix[s] >= lu - big_m * is_)

                # suffix_sum == 0  →  ie_ = 1
                self.model.addConstr(suffix[s] <= big_m * (1 - ie_))
                self.model.addConstr(suffix[s] >= lu - big_m * ie_)

                # If the link is never used (lu==0) → force is_=1 and ie_=0 for correct idle_lat calculation
                self.model.addConstr(is_ >= 1 - lu)
                self.model.addConstr(ie_ <= lu)

        # ------------------------------------------------------------------
        # 2) idle-latency per resource
        # ------------------------------------------------------------------
        self.idle_lat: dict[Resource, gp.Var] = {}
        for res in {r for r, _ in self.idleS} | {r for r, _ in self.idleE}:
            expr = quicksum(
                self.idleS.get((res, s), 0) * self.slot_latency[s] + self.idleE.get((res, s), 0) * self.slot_latency[s]
                for s in range(max_s + 1)
            )
            v = self.model.addVar(vtype=GRB.INTEGER, name=f"idleLat_{_resource_key(res)}")
            self.model.addConstr(v == expr)
            self.idle_lat[res] = v

        # ------------------------------------------------------------------
        # 3) overlap  =  min idle_lat[r]   (only over resources that are used)
        # ------------------------------------------------------------------
        overlap = self.model.addVar(vtype=GRB.INTEGER, name="overlap")
        self.overlap = overlap
        for v in self.idle_lat.values():
            self.model.addConstr(overlap <= v)

        # ------------------------------------------------------------------
        # EXTRA) transfer fires (across all transfers and steady-state iterations)
        # ------------------------------------------------------------------
        transfer_fires = quicksum(self.fires[tr] for tr in self.transfer_nodes)

        # ------------------------------------------------------------------
        # 4) total latency + objective
        # ------------------------------------------------------------------
        total_lat = self.model.addVar(vtype=GRB.INTEGER, name="total_latency")
        self.total_latency = total_lat
        self.model.addConstr(
            total_lat == self.iterations * quicksum(self.slot_latency.values()) - (self.iterations - 1) * overlap
        )
        obj_func = total_lat + transfer_fires
        self.model.setObjective(obj_func, GRB.MINIMIZE)

    # ------------------------------------------------------------------ #
    # public solve()                                                     #
    # ------------------------------------------------------------------ #
    def solve(self, *, tee: bool = True) -> tuple[TimeSlotAllocation, SteadyStateWorkload, int]:  # noqa: PLR0912
        self.model.setParam("OutputFlag", 1 if tee else 0)
        self.model.optimize()
        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError("Gurobi did not find an optimal solution.")

        # ---------- read back decisions --------------------------------
        var_threshold = 0.5  # threshold to decide if a variable is chosen
        tensor_alloc = self.get_tensor_allocations(var_threshold)
        routing = self.get_transfer_routing(var_threshold)
        self.update_transfer_reuse_levels(var_threshold)

        # ---------- rebuild TSA with new resources ---------------------
        new_allocs: list[tuple[int, Resource, SteadyStateNode]] = []
        for slot, res, node in self.tsa.allocations:
            if isinstance(node, SteadyStateTensor):
                if res is None:
                    assert node in tensor_alloc, f"Tensor {node.node_name} not found in tensor_alloc."
                    res_new = tensor_alloc[node]
                else:
                    res_new = res
                new_allocs.append((slot, res_new, node))
            elif isinstance(node, SteadyStateTransfer):
                assert node in routing, f"Transfer {node.node_name} not found in routing."
                res_new = routing[node]
                for link in res_new:
                    new_allocs.append((slot, link, node))
            elif isinstance(node, SteadyStateComputation):
                assert res is not None, f"Computation {node.node_name} has no resource assigned."
                new_allocs.append((slot, res, node))
            else:
                raise TypeError(f"Unexpected node type: {type(node)} in TSA allocations.")
        # Update the TimeSlotAllocation with the new allocations
        tsa_upd = TimeSlotAllocation(new_allocs)
        # Update the steady state workload to prevent graph inconsistencies
        ssw_upd = self.get_updated_steady_state_workload()
        assert self.total_latency is not None, "Total latency variable was not created."
        return tsa_upd, ssw_upd, int(self.total_latency.X)

    def update_transfer_reuse_levels(self, var_threshold: float) -> None:
        reuse_levels: dict[SteadyStateTransfer, int] = {}
        for tr in self.transfer_nodes:
            for stop in range(-1, len(tr.steady_state_iteration_space)):
                if self.z_cache[(tr, stop)].X > var_threshold:
                    reuse_levels[tr] = stop
                    break
            else:
                raise ValueError(f"{tr.node_name}: no reuse level chosen.")
        # Update the steady state iteration space for transfers
        for tr in self.transfer_nodes:
            stop = reuse_levels[tr]
            # Set all iteration variables to REUSE flag
            for i, iter_var in enumerate(tr.steady_state_iteration_space):
                if i <= stop:
                    flag = IterationVariableReuse.REUSE
                else:
                    flag = IterationVariableReuse.NO_REUSE
                iter_var.reuse = flag

    def get_transfer_routing(self, var_threshold: float) -> dict[SteadyStateTransfer, tuple[CommunicationLink]]:
        routing: dict[SteadyStateTransfer, tuple[CommunicationLink]] = {}
        for tr in self.transfer_nodes:
            chosen = [p for p in tr.possible_resource_allocation if self.y_path[(tr, tuple(p))].X > var_threshold]
            if len(chosen) != 1:
                raise ValueError(f"{tr.node_name}: expected exactly one path, got {chosen}")
            path = chosen[0]
            tr.chosen_resource_allocation = path
            tr.runtime = self._transfer_latency(tr, path)
            routing[tr] = path
        return routing

    def get_tensor_allocations(self, var_threshold: float) -> dict[SteadyStateTensor, Core]:
        tensor_alloc: dict[SteadyStateTensor, Core] = {}
        for t in self.tensor_var:
            chosen = [c for c in t.possible_resource_allocation if self.x_tensor[(t, c)].X > var_threshold]
            if len(chosen) != 1:
                raise ValueError(f"{t.node_name}: expected exactly one core, got {chosen}")
            core = chosen[0]
            t.chosen_resource_allocation = core
            tensor_alloc[t] = core
        return tensor_alloc

    def get_updated_steady_state_workload(self) -> SteadyStateWorkload:
        """Return a new SteadyStateWorkload with updated resource allocations.
        This is necessary as there's a bug in networkx that doesn't update the edges when node attributes change.
        """
        ssw_upd = SteadyStateWorkload()
        for edge in self.ssw.edges(data=True):
            src, dst, data = edge
            src = next(n for n in self.ssw.node_list if n is src)
            dst = next(n for n in self.ssw.node_list if n is dst)
            ssw_upd.add_edge(src, dst, **data)
        return ssw_upd
