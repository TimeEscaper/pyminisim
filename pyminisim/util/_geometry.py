import numpy as np


def closest_point_of_segment(point: np.ndarray, segment: np.ndarray) -> np.ndarray:
    assert point.shape == (2,) or (len(point.shape) == 2 and point.shape[1] == 2), \
        f"Point must have shape (N, 2) or (2,), got {point.shape}"
    assert segment.shape == (4,) or (len(segment.shape) == 2 and segment.shape[1] == 4), \
        f"Segment must have shape (4,) or (M, 4), got {segment.shape}"

    if len(point.shape) == 2:
        single_point = False
    else:
        point = point[np.newaxis]
        single_point = True
    if len(segment.shape) == 2:
        single_segment = False
    else:
        segment = segment[np.newaxis]
        single_segment = True

    p1 = segment[:, :2]
    p2 = segment[:, 2:]
    p1p2 = p2 - p1
    o1p1 = point[:, np.newaxis, :] - p1[np.newaxis, :, :]
    dot_1 = np.einsum("ijk,jk->ij", o1p1, p1p2)  # (N, M)
    dot_2 = np.einsum("ij,ij->i", p1p2, p1p2)  # (M,)
    t = dot_1 / dot_2
    t = np.clip(t, 0, 1)  # (N, M)
    p3 = segment[np.newaxis, :, :2] + t[:, :, np.newaxis] * p1p2[np.newaxis, :, :]  # (N, M, 2)

    if single_segment:
        p3 = p3[:, 0, :]
    if single_point:
        p3 = p3[0]

    return p3


def point_to_segment_distance(point: np.ndarray, segment: np.ndarray) -> float:
    p3 = closest_point_of_segment(point, segment)
    distance = np.linalg.norm(point - p3, axis=-1)
    return distance
