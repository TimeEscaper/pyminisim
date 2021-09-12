from typing import List, Union

from pyminisim.measures.point import Point


class Trajectory:

    def __init__(self, waypoints: List[Point]):
        self._waypoints = waypoints.copy()

    @property
    def waypoints(self) -> List[Point]:
        return self._waypoints.copy()

    @waypoints.setter
    def waypoints(self, value: List[Point]):
        self._waypoints = value.copy()

    @property
    def current_waypoint(self) -> Union[Point, None]:
        if len(self._waypoints) != 0:
            return self._waypoints[0]
        else:
            return None

    def transit(self) -> Union[Point, None]:
        if len(self._waypoints) != 0:
            return self._waypoints.pop(0)
        else:
            return None
