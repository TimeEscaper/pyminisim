import numpy as np


def point_to_segment_distance(point: np.ndarray, segment: np.ndarray) -> float:
    assert point.shape == (2,)
    assert segment.shape == (4,)
    p1 = segment[:2]
    p2 = segment[2:]
    # Find projection
    p1p2 = p2 - p1
    o1p1 = point - p1
    t = (np.dot(o1p1, p1p2) / np.dot(p1p2, p1p2))
    t = np.clip(t, 0, 1)
    x3 = p1[0] + p1p2[0] * t
    y3 = p1[1] + p1p2[1] * t
    p3 = np.array([x3, y3])
    distance = np.linalg.norm(point - p3)
    return distance
