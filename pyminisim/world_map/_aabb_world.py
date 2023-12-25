from typing import Optional, Union, Tuple, List

import numpy as np
from scipy.spatial.distance import cdist

from pyminisim.core import AbstractWorldMapState, AbstractStaticWorldMap
from pyminisim.util import point_to_segment_distance


class _AABBWorldState(AbstractWorldMapState):

    def __init__(self, boxes: np.ndarray,
                 colors: Optional[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]] = None):
        # boxes: (N, dim); dim = { tl_x, tl_y, width, height }
        assert len(boxes.shape) == 2 and boxes.shape[1] == 4, \
            f"Boxes must have shape [N, dim=4 (tl_x, tl_y, width, height)], got {boxes.shape}"
        if colors is not None:
            if isinstance(colors, list):
                assert len(colors) == boxes.shape[0], \
                    (f"For multiple colors, number of colors must match number of boxes, "
                     f"got {len(colors)} colors and {boxes.shape[0]} boxes")
                for i, color in enumerate(colors):
                    assert len(color) == 3, f"All colors must be tuples (int, int, int), got {color} for color {i}"
            else:
                assert len(colors) == 3, f"Color must be tuple (int, int, int), got {colors}"
        super(_AABBWorldState, self).__init__()
        self._boxes = boxes
        self._colors = colors
        segments = []
        for box in boxes:
            segments.append([box[0], box[1], box[0] + box[2], box[1]])
            segments.append([box[0], box[1], box[0], box[1] + box[3]])
            segments.append([box[0] + box[2], box[1], box[0] + box[2], box[1] + box[3]])
            segments.append([box[0], box[1] + box[3], box[0] + box[2], box[1] + box[3]])
        self._segments = np.array(segments)

    @property
    def boxes(self) -> np.ndarray:
        return self._boxes

    @property
    def colors(self) -> Optional[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]:
        return self._colors

    @property
    def segments(self) -> np.ndarray:
        return self._segments


class AABBWorld(AbstractStaticWorldMap):

    def __init__(self,
                 boxes: np.ndarray,
                 colors: Optional[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]] = None):
        """
        :param boxes: List of circles in format [[tl_x, tl_y, width, height]] (in metres)
        """
        super(AABBWorld, self).__init__(_AABBWorldState(boxes.copy(), colors))

    def closest_distance_to_obstacle(self, point: np.ndarray) -> \
            Union[float, np.ndarray]:
        assert point.shape == (2,) or (len(point.shape) == 2 and point.shape[1] == 2)

        if point.shape == (2,):
            point = point[np.newaxis, :]
            single_input = True
        else:
            single_input = False

        segments = self.current_state.segments
        distances = []
        for i in range(point.shape[0]):
            point_distance = np.inf
            for j in range(segments.shape[0]):
                distance = point_to_segment_distance(point[i], segments[j])
                if distance < point_distance:
                    point_distance = distance
            distances.append(point_distance)

        if single_input:
            return distances[0]
        return np.array(distances)

    def is_occupied(self, point: np.ndarray) -> Union[bool, np.ndarray]:
        assert point.shape == (2,) or (len(point.shape) == 2 and point.shape[1] == 2)

        if point.shape == (2,):
            point = point[np.newaxis, :]
            single_input = True
        else:
            single_input = False

        boxes = self.current_state.boxes
        result = []
        for i in range(point.shape[0]):
            occupied = False
            for j in range(boxes.shape[0]):
                tl_x, tl_y, w, h = boxes[j]
                x, y = point[i]
                occupied = (tl_x <= x <= (tl_x + w)) and (tl_y <= y <= (tl_y + h))
                if occupied:
                    break
            result.append(occupied)

        if single_input:
            return result[0]
        return np.array(result)

    @property
    def boxes(self) -> np.ndarray:
        return self._state.boxes.copy()

    @property
    def colors(self) -> Optional[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]:
        return self._state.colors
