"""Display order derived from the registered acquisition, independent of analysis rankings."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from .models import ChannelSpec, RecordingSpec, parse_shaft_group


def _contact(channel: ChannelSpec) -> tuple[str, int] | None:
    if channel.is_auxiliary or channel.kind != "seeg" or channel.shaft in {"AUX", "OTHER"}:
        return None
    # For an acquired bipolar pair, only two contacts from the same shaft
    # identify a depth lead. FP1-F7, for example, is not a shaft called FP1-F.
    first, sep, second = channel.name.partition("-")
    if sep:
        left = parse_shaft_group(first, channel.kind)
        if left["shaft"] not in {"AUX", "OTHER"}:
            right = parse_shaft_group(second, channel.kind)
            if left["shaft"] != right["shaft"]:
                return None
            return left["shaft"], left["contact_index"]
    if channel.contact_index >= 9999:
        return None
    return channel.shaft, channel.contact_index


class ChannelDisplayOrder:
    def __init__(self, recording: RecordingSpec):
        channels = recording.channels
        self._id_by_name = {ch.name: ch.id for ch in channels}
        self._positions = {ch.id: i for i, ch in enumerate(channels)}
        self._shafts: dict[str, str] = {}
        if recording.modality == "scalp":
            return

        groups: dict[str, set[int]] = defaultdict(set)
        contacts = {}
        for ch in channels:
            contact = _contact(ch)
            if contact is not None:
                groups[contact[0]].add(contact[1])
                contacts[ch.id] = contact
        depth = [cid for cid, (shaft, _) in contacts.items() if len(groups[shaft]) >= 2]
        self._shafts = {cid: contacts[cid][0] for cid in depth}
        # Move only depth contacts. Other leads keep their acquisition slots.
        slots = [self._positions[cid] for cid in depth]
        self._positions.update(zip(sorted(depth, key=contacts.__getitem__), slots))

    def _source(self, cid: str, name: str, parents: Sequence[str]) -> str:
        if cid in self._positions:
            return cid
        # Average-reference outputs retain the primary channel's name. A
        # bipolar output takes its position from its first (positive) parent.
        named = self._id_by_name.get(name)
        if named is not None and (not parents or named in parents):
            return named
        return parents[0] if parents else cid

    def position(self, cid: str, name: str, parents: Sequence[str] = ()) -> int:
        return self._positions.get(self._source(cid, name, parents), len(self._positions))

    def row_position(self, row: dict[str, Any]) -> int:
        return self.position(row.get("channel_id", ""), row["channel_name"], row.get("parents") or ())

    def depth_shaft(self, cid: str, name: str, parents: Sequence[str] = ()) -> str | None:
        source = self._source(cid, name, parents)
        shaft = self._shafts.get(source)
        if cid not in self._positions and parents and source != self._id_by_name.get(name):
            # A cross-shaft bipolar pair has no single depth shaft. A common-
            # average output retains the name and location of the source lead.
            if any(self._shafts.get(parent) != shaft for parent in parents):
                return None
        return shaft
