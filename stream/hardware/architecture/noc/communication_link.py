from typing import TYPE_CHECKING, Literal

import numpy as np

from stream.cost_model.communication_manager import CommunicationLinkEvent

if TYPE_CHECKING:
    from stream.hardware.architecture.core import Core
    from stream.workload.tensor import SubviewTensor

ENABLE_BROADCASTING = False


def get_bidirectional_edges(
    core_a: "Core",
    core_b: "Core",
    bandwidth: float,
    unit_energy_cost: float,
    link_type: Literal["bus"] | Literal["link"],
) -> list[tuple["Core", "Core", dict[str, "CommunicationLink"]]]:
    """Create a list with two edges: from A to B and B to A."""
    bus = CommunicationLink("Any", "Any", bandwidth, unit_energy_cost)
    link_a_to_b = CommunicationLink(core_a, core_b, bandwidth, unit_energy_cost)
    link_b_to_a = CommunicationLink(core_b, core_a, bandwidth, unit_energy_cost)

    # if have_shared_memory(core_a, core_b):
    #     # No edge if the cores have a shared memory
    #     return []

    return [
        #  A -> B
        (
            core_a,
            core_b,
            {"cl": bus if link_type == "bus" else link_a_to_b},
        ),
        # B -> A
        (
            core_b,
            core_a,
            {"cl": bus if link_type == "bus" else link_b_to_a},
        ),
    ]


class CommunicationLink:
    """Represents a fixed-bandwidth communication link used to communicate between two cores."""

    def __init__(
        self,
        sender: "Core | Literal['Any']",
        receiver: "Core | Literal['Any']",
        bandwidth: int | float,
        unit_energy_cost: float,
        bidirectional: bool = False,
    ) -> None:
        self.sender = sender
        self.receiver = receiver
        self.bandwidth = bandwidth
        self.unit_energy_cost = unit_energy_cost
        self.bidirectional = bidirectional  # TODO this property is not in use?

        self.events: list[CommunicationLinkEvent] = []
        self.active_periods = [(0, float("inf"), 0)]
        self.active_ts = np.array([0, float("inf")])
        self.active_deltas = np.array([0, 0])
        self.tensors: dict[SubviewTensor, list[CommunicationLinkEvent]] = {}

    def __str__(self) -> str:
        return f"CommunicationLink({self.sender}, {self.receiver}, bw={self.bandwidth})"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash((self.sender, self.receiver, self.bandwidth, self.unit_energy_cost, self.bidirectional))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CommunicationLink) and (self.sender, self.receiver, self.bandwidth) == (
            other.sender,
            other.receiver,
            other.bandwidth,
        )

    def get_name_for_schedule_plot(self) -> str:
        if self.bidirectional:
            return f"{self.sender} <-> {self.receiver}"
        else:
            return f"{self.sender} -> {self.receiver}"

    def transfer(self, link_event: CommunicationLinkEvent) -> float:
        """Transfer data on this communication link at timestep.
        The transfer can take longer than necessary for this link if another lower-bandwidth link is involved.

        Args:
            tensor : The tensor to be transferred.
            start : The timestep in clock cyles to start the transfer.
            duration : The duration of the transfer.

        Returns:
            int: The end time when communication on this link is finished
        """
        energy_cost = link_event.energy
        self.update_activity(link_event)
        return energy_cost

    def block(
        self,
        start: int,
        duration: int,
        tensors: list["SubviewTensor"],
        activity: int = 100,
        source: "Core" = None,
        destinations: "Core" = None,
    ):
        """Block this communication link from start timestep for a given duration.

        Args:
            start: The timestep at which the blocking starts.
            duration: The duration of the blocking.
            tensors: A list of tensors for which we are blocking the link.
            activity: The percentage of the link bandwidth used
        """
        end = start + duration
        # Create a CLEvent
        event = CommunicationLinkEvent(
            type="block",
            start=start,
            end=end,
            tensors=tensors,
            energy=tensors[0].cn_source.get_offchip_energy(),
            activity=activity,
            source=source,
            destinations=destinations,
        )
        self.update_activity(event)
        return

    def update_activity(self, event: CommunicationLinkEvent):
        start = event.start
        end = event.end
        activity = event.activity
        if start == end:
            return
        # Check if this is a duplicate event for broadcast
        for tensor in event.tensors:
            previous_events = self.tensors.get(tensor, [])
            for previous_event in previous_events:
                if previous_event.start == event.start and previous_event.end == event.end:
                    # Update the previous event's destinations with this event's destinations
                    return
        idx_start = np.searchsorted(self.active_ts, start)
        if self.active_ts[idx_start] == start:
            self.active_deltas[idx_start] += activity
        else:
            self.active_ts = np.insert(self.active_ts, idx_start, start)
            self.active_deltas = np.insert(self.active_deltas, idx_start, activity)
        idx_end = np.searchsorted(self.active_ts, end)
        if self.active_ts[idx_end] == end:
            self.active_deltas[idx_end] -= activity
        else:
            self.active_ts = np.insert(self.active_ts, idx_end, end)
            self.active_deltas = np.insert(self.active_deltas, idx_end, -activity)
        # Track that this link has transferred the tensors of this event for future broadcasts
        for tensor in event.tensors:
            self.tensors[tensor] = self.tensors.get(tensor, []) + [event]
        self.events.append(event)

    def get_idle_window(self, activity: float, duration: int, earliest_t: int, tensors: list["SubviewTensor"]):
        """
        Get the earliest time window of duration `duration` from `earliest_t` with at least `activity` percent
        available.
        """
        valid_windows: list[tuple[int, int]] = []
        ## Check if this tensor has already been transferred on this link before
        if ENABLE_BROADCASTING:
            # If so, check duration and earliest timestep requirements of this call
            for tensor in tensors:
                if tensor in self.tensors:
                    previous_events = self.tensors[tensor]
                    # Get the latest valid previous event
                    duration_and_earliest_t_valid_previous_events = [
                        previous_event
                        for previous_event in previous_events
                        if previous_event.start >= earliest_t and previous_event.duration >= duration
                    ]
                    if duration_and_earliest_t_valid_previous_events:
                        # Add the latest valid previous event to the list of valid windows
                        previous_valid_event = duration_and_earliest_t_valid_previous_events[-1]
                        valid_windows.append((previous_valid_event.start, previous_valid_event.end))
        ## Check other possible periods given the activity
        activities = np.cumsum(self.active_deltas)
        earliest_t_index = np.searchsorted(self.active_ts, earliest_t, side="right")
        relevant_ts = self.active_ts[earliest_t_index:]
        updated_ts = relevant_ts.copy()
        relevant_activities = activities[earliest_t_index:]
        # Insert the earliest timestep and the activity at that timestep
        updated_ts = np.insert(updated_ts, 0, earliest_t)
        updated_activities = np.insert(relevant_activities, 0, activities[earliest_t_index - 1])
        updated_activities = updated_activities + activity
        idxs = np.argwhere(updated_activities > self.bandwidth)
        idxs = [idx[0] for idx in idxs]
        idxs.append(len(updated_ts) - 1)
        start = earliest_t
        for idx in idxs:
            end: int = updated_ts[idx]
            if end - start >= duration:
                valid_windows.append((start, end))
            try:
                start: int = updated_ts[idx + 1]
            except IndexError:
                break
        if not valid_windows:
            raise ValueError(f"There are no valid windows of activity {activity} and duration {duration} for {self}.")
        return valid_windows
