from abc import ABC, abstractmethod
import numpy as np
from typing import List

from pyminisim.measures.point import Point


class Shape(ABC):

    @abstractmethod
    def get_closest_point(self, target: Point) -> Point:
        raise NotImplementedError()


class PolygonalShape(Shape, ABC):

    @abstractmethod
    def get_segments(self) -> np.ndarray:
        raise NotImplementedError()


class Segment(PolygonalShape):

    def __init__(self, start_point: Point, end_point: Point):
        super(PolygonalShape, self).__init__()
        self._start_point = start_point
        self._end_point = end_point

    @property
    def start(self) -> Point:
        return self._start_point

    @property
    def end(self) -> Point:
        return self._end_point

    def get_closest_point(self, target: Point):
        end_point_vec = self._end_point.vector - self._start_point.vector
        target_point_vec = target.vector - self._start_point.vector
        alpha = (end_point_vec @ target_point_vec) / (np.linalg.norm(target_point_vec) ** 2)
        if alpha <= 0:
            return Point(self._start_point.vector)
        elif alpha >= 1:
            return Point(self._end_point.vector)
        else:
            return Point(self._start_point.vector + alpha * end_point_vec)

    def get_segments(self) -> np.ndarray:
        return np.array([[self._start_point.x, self._end_point.x, self._start_point.y, self._end_point.y]])


class Box(PolygonalShape):

    def __init__(self, tl: Point, tr: Point, br: Point, bl: Point):
        self._tl = tl
        self._tr = tr
        self._br = br
        self._bl = bl

    @property
    def tl(self) -> Point:
        return self._tl

    @property
    def tr(self) -> Point:
        return self._tr

    @property
    def br(self) -> Point:
        return self._br

    @property
    def bl(self) -> Point:
        return self._bl

    def get_closest_point(self, target: Point) -> Point:
        segment_points = [segment.get_closest_point(target) for segment in self._to_segments()]
        return segment_points[np.argmin([segment_point.distance(target) for segment_point in segment_points])]

    def get_segments(self) -> np.ndarray:
        return np.concatenate([segment.get_segments() for segment in self._to_segments()], axis=1)

    def _to_segments(self) -> List[Segment]:
        return [Segment(self._tl, self._tr),
                Segment(self._tl, self._bl),
                Segment(self._tr, self._br),
                Segment(self._br, self._bl)]
