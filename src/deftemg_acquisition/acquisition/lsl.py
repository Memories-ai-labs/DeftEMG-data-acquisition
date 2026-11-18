"""Lab Streaming Layer (LSL) helpers shared by every device streamed live.

Each device's live streams (the armband's ``armband.live``, the glove's ``wuji``) are named
with ``STREAM_PREFIX`` and share the session's ``source_id``, so viewers can find them and
match them up.
"""

from __future__ import annotations

import pylsl

STREAM_PREFIX = "Waveletech"


def make_outlet(
    name, stype, channels, units, rate, fmt, source_id, manufacturer="WAVELETECH", **desc: str
) -> pylsl.StreamOutlet:
    """An LSL outlet with channel labels and units. Another stream joins the session's by
    sharing its ``source_id`` (``LiveStreamer.source_id``), named with ``STREAM_PREFIX``."""
    info = pylsl.StreamInfo(name, stype, len(channels), rate, fmt, source_id)
    info.desc().append_child_value("manufacturer", manufacturer)
    for key, value in desc.items():
        info.desc().append_child_value(key, value)
    chans = info.desc().append_child("channels")
    for label, unit in zip(channels, units, strict=True):
        ch = chans.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", unit)
    return pylsl.StreamOutlet(info)
