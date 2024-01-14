from typing import List
from shapely.geometry import LineString as ShapelyLine
from pandas import Interval


class TimedEdge:
    """
    Represents a timed edge in a graph.

    Attributes:
        geometry (ShapelyLine): The geometry of the edge.
        availability (List[Interval]): The availability intervals of the edge.

    Methods:
        __init__(self, geometry: ShapelyLine, availability: List[Interval]):
            Initialize a TimedEdge object.

        # TODO - add collision checking functions for given time interval / time instance
    """

    def __init__(
        self,
        geometry: ShapelyLine,
        always_available: bool,
        availability: List[Interval],
    ):
        """
        Initialize a TimedEdge object.

        Args:
            geometry (ShapelyLine): The geometry of the edge.
            always_available (bool): Whether the edge is always available.
            availability (List[Interval]): The availability intervals of the edge.
        """
        self.geometry = geometry
        self.always_available = always_available
        self.availability = availability