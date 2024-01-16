import numpy as np

from typing import Optional, Union, Tuple, List, Dict
from dataclasses import dataclass
from pyminisim.core import AbstractWorldMapState, AbstractStaticWorldMap
from pyminisim.util import point_to_segment_distance, closest_point_of_segment


@dataclass
class AABBObject:
    box: Tuple[float, float, float, float]
    class_name: Optional[str] = None
    object_name: Optional[str] = None
    color: Optional[Tuple[int, int, int]] = None


class _AABBWorldState(AbstractWorldMapState):

    def __init__(self, objects: List[AABBObject], default_color: Tuple[int, int, int]):
        # boxes: (N, dim); dim = { tl_x, tl_y, width, height }
        boxes = []
        segments = []
        colors = []
        semantic_objects = {}
        for i, obj in enumerate(objects):
            assert len(obj.box) == 4, f"Object box must be a tuple of 4 floats, got {obj.box} for object {i}"
            if obj.color is not None:
                assert len(obj.color) == 3, f"Object color must be a tuple of 3 ints from 0 to 255, got {obj.color}"
            box = obj.box
            boxes.append(box)
            tl_x, tl_y, w, h = box
            segments.append([tl_x, tl_y, tl_x, tl_y + w])
            segments.append([tl_x - h, tl_y, tl_x - h, tl_y + w])
            segments.append([tl_x, tl_y, tl_x - h, tl_y])
            segments.append([tl_x, tl_y + w, tl_x - h, tl_y + w])
            # segments.append([box[0], box[1], box[0] + box[2], box[1]])
            # segments.append([box[0], box[1], box[0], box[1] + box[3]])
            # segments.append([box[0] + box[2], box[1], box[0] + box[2], box[1] + box[3]])
            # segments.append([box[0], box[1] + box[3], box[0] + box[2], box[1] + box[3]])
            colors.append(obj.color if obj.color is not None else default_color)
            if obj.object_name is not None and obj.class_name is None:
                raise ValueError(f"class_name must be specified if object_name is not None, violated for object at {i}")
            if obj.class_name is not None:
                semantic_objects[i] = (obj.class_name, obj.object_name)

        self._boxes = np.array(boxes)
        self._segments = np.array(segments)
        self._colors = colors
        self._semantic_objects = semantic_objects

        # assert len(boxes.shape) == 2 and boxes.shape[1] == 4, \
        #     f"Boxes must have shape [N, dim=4 (tl_x, tl_y, width, height)], got {boxes.shape}"
        # if colors is not None:
        #     if isinstance(colors, list):
        #         assert len(colors) == boxes.shape[0], \
        #             (f"For multiple colors, number of colors must match number of boxes, "
        #              f"got {len(colors)} colors and {boxes.shape[0]} boxes")
        #         for i, color in enumerate(colors):
        #             assert len(color) == 3, f"All colors must be tuples (int, int, int), got {color} for color {i}"
        #     else:
        #         assert len(colors) == 3, f"Color must be tuple (int, int, int), got {colors}"
        # super(_AABBWorldState, self).__init__()
        # self._boxes = boxes
        # self._colors = colors
        # segments = []
        # for box in boxes:
        #     segments.append([box[0], box[1], box[0] + box[2], box[1]])
        #     segments.append([box[0], box[1], box[0], box[1] + box[3]])
        #     segments.append([box[0] + box[2], box[1], box[0] + box[2], box[1] + box[3]])
        #     segments.append([box[0], box[1] + box[3], box[0] + box[2], box[1] + box[3]])
        # self._segments = np.array(segments)

    @property
    def boxes(self) -> np.ndarray:
        return self._boxes

    @property
    def colors(self) -> Optional[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]:
        return self._colors

    @property
    def segments(self) -> np.ndarray:
        return self._segments

    @property
    def semantic_objects(self) -> Dict[int, Tuple[str, str]]:
        return self._semantic_objects


class AABBWorld(AbstractStaticWorldMap):

    def __init__(self, objects: List[AABBObject], default_color: Optional[Tuple[int, int, int]] = (0, 255, 0)):
        """
        :param boxes: List of circles in format [[tl_x, tl_y, width, height]] (in metres)
        """
        super(AABBWorld, self).__init__(_AABBWorldState(objects, default_color))

    def closest_distance_to_obstacle(self, point: np.ndarray) -> \
            Union[float, np.ndarray]:
        assert point.shape == (2,) or (len(point.shape) == 2 and point.shape[1] == 2), \
            f"point must have shape (2,) or (N, 2), got {point.shape}"
        if point.shape == (2,):
            point = point[np.newaxis, :]
            single_input = True
        else:
            single_input = False

        all_distances = point_to_segment_distance(point, self.current_state.segments)
        closest_distances = np.min(all_distances, axis=1)
        if single_input:
            closest_distances = closest_distances[0]

        return closest_distances

        # if point.shape == (2,):
        #     point = point[np.newaxis, :]
        #     single_input = True
        # else:
        #     single_input = False
        #
        # segments = self.current_state.segments
        # distances = []
        # for i in range(point.shape[0]):
        #     point_distance = np.inf
        #     for j in range(segments.shape[0]):
        #         distance = point_to_segment_distance(point[i], segments[j])
        #         if distance < point_distance:
        #             point_distance = distance
        #     distances.append(point_distance)
        #
        # if single_input:
        #     return distances[0]
        # return np.array(distances)

    def is_occupied(self, point: np.ndarray) -> Union[bool, np.ndarray]:
        assert point.shape == (2,) or (len(point.shape) == 2 and point.shape[1] == 2)

        if point.shape == (2,):
            point = point[np.newaxis, :]
            single_input = True
        else:
            single_input = False

        boxes = self.current_state.boxes
        result = np.zeros((point.shape[0]), dtype=bool)
        for box in boxes:
            tl_x, tl_y, w, h = box
            occupied = np.logical_and(
                np.logical_and(point[:, 0] <= tl_x, point[:, 0] >= (tl_x - h)),
                np.logical_and(point[:, 1] >= tl_y, point[:, 1] <= (tl_y + w))
            )
            result = np.logical_or(result, occupied)

        if single_input:
            return result[0]
        return np.array(result)

    @property
    def boxes(self) -> np.ndarray:
        return self._state.boxes.copy()

    @property
    def colors(self) -> Optional[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]:
        return self._state.colors

    def closest_semantic_objects(self,
                                 point: np.ndarray,
                                 max_distance: float) -> Dict[int, Tuple[str, str, np.ndarray]]:
        # result = {}
        # semantic_objects = self._state.semantic_objects
        # if len(semantic_objects) == 0:
        #     return result
        #
        # segments = self._state.segments[semantic_objects.keys()]

        result = {}
        for obj_idx, object_info in self._state.semantic_objects.items():
            segments = self._state.segments[4 * obj_idx:4 * (obj_idx+1)]
            # closest_points = np.stack([closest_point_of_segment(point, segments[i]) for i in range(4)], axis=0)
            closest_points = closest_point_of_segment(point, segments)
            distances = np.linalg.norm(point - closest_points, axis=1)
            closest_idx = np.argmin(distances)
            if distances[closest_idx] <= max_distance:
                closest_point = closest_points[closest_idx]
                result[obj_idx] = object_info + (closest_point,)
        return result
