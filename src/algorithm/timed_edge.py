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

        is_available(self, query_interval: Interval) -> bool:
            Check if the edge is available for the given time interval.

        __covers_interval(self, interval: Interval, other: Interval) -> bool:
            Check if the edge covers the given interval.
    """

    def __init__(
        self,
        geometry: ShapelyLine,
        availability: List[Interval],
        always_available: bool = False,
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

    def is_available(self, query_interval: Interval) -> bool:
        """
        Check if the edge is available for the given time interval.

        Args:
            query_interval (Interval): The time interval to check.

        Returns:
            bool: True if the edge is available, False otherwise.
        """
        if self.always_available:
            return True

        if len(self.availability) == 1:
            return self.__covers_interval(self.availability[0], query_interval)

        # perform binary search to find the first interval intersecting with the query_interval
        left = 0
        right = len(self.availability) - 1

        while left <= right:
            # only one interval is left from the search that potentially covers the query_interval
            if left == right:
                return self.__covers_interval(self.availability[left], query_interval)

            # select the center interval
            mid = (left + right) // 2

            # if the query interval ends before the middle interval starts, search left
            if self.availability[mid].left > query_interval.right:
                right = mid - 1
                continue

            # if the query interval starts after the current interval ends, search right
            elif self.availability[mid].right < query_interval.left:
                left = mid + 1
                continue

            # query interval ends after the current interval starts and starts before the current interval ends
            # -> the query interval at least partially lies within the current interval
            else:
                return self.__covers_interval(self.availability[mid], query_interval)

        return False

    def __covers_interval(self, interval: Interval, other: Interval) -> bool:
        """
        Check if the edge covers the given interval.

        Args:
            interval (Interval): The interval to check.
            other (Interval): The interval that should be covered.

        Returns:
            bool: True if the edge covers the interval, False otherwise.
        """
        return interval.left <= other.left and interval.right >= other.right
