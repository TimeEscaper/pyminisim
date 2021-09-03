from pyminisim.measures.shape import Shape


class SimObstacle:

    def __init__(self, obstacle_id: int, shape: Shape, obstacle_type: str, name: str = ""):
        self._obstacle_id = obstacle_id
        self._shape = shape
        self._obstacle_type = obstacle_type
        self._obstacle_name = name

    @property
    def obstacle_id(self) -> int:
        return self._obstacle_id

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def obstacle_type(self) -> str:
        return self._obstacle_type

    @property
    def obstacle_name(self) -> str:
        return self._obstacle_name
