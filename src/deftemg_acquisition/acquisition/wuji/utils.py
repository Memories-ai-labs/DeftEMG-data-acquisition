from typing import Any
import numpy as np


def _quat(q) -> list[float]:
    return [q.x, q.y, q.z, q.w]


def _vec(v) -> list[float]:
    return [v.x, v.y, v.z]


def _data(frame) -> dict[str, Any]:
    return {"data": frame.data}


def _zones(frame) -> dict[str, Any]:
    return {z: getattr(frame, z) for z in ("palm", "thumb", "index", "middle", "ring", "pinky")}


def _joint_angles(frame) -> dict[str, Any]:
    # firmware order, fingers thumb..pinky, each finger's angles concatenated
    return {
        "angles": [a for finger in frame.fingers for a in finger.angles],
        "confidence": [finger.confidence for finger in frame.fingers],
    }