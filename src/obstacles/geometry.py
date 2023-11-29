from shapely.geometry import Point, LineString, Polygon
from pandas import Interval


class Geometry:
    """
    Represents a geometric object with a radius and time interval.

    Attributes:
        radius (float): The radius around the point, considered to be in collision.
        interval (Interval): The closed time interval.

    Methods:
        set_interval(lower_bound: float, upper_bound: float): Sets the closed time interval.
        set_radius(radius: float): Sets the radius around the point.
        check_collision(shape, query_time=None, query_interval=None): Checks if the point is in collision with a given shape.
        is_active(query_time=None, query_interval=None): Checks if the geometry is active at a given time or time interval.
    """

    def __init__(self, radius: float = None, interval: Interval = None):
        """
        Initializes a Geometry object with the given radius and time interval.

        Args:
            radius (float): The radius around the point, considered to be in collision.
            interval (Interval): The closed time interval.
        """
        self.radius = radius
        self.time_interval = interval

    def set_interval(self, lower_bound: float, upper_bound: float):
        """
        Sets the closed time interval from a lower and upper bound.

        Args:
            lower_bound (float): The lower bound of the interval.
            upper_bound (float): The upper bound of the interval.
        """
        if lower_bound > upper_bound:
            raise ValueError("The lower bound must be smaller than the upper bound.")

        self.time_interval = Interval(lower_bound, upper_bound, closed="both")

    def set_radius(self, radius: float):
        """
        Sets the radius around the point, considered to be in collision.

        Args:
            radius (float): The radius around the point.
        """
        self.radius = radius

    def is_active(self, query_time: float = None, query_interval: Interval = None):
        """
        Checks if the geometry is active at a given time or time interval.

        Args:
            query_time (optional): The specific time to check activity at.
            query_interval (optional): The time interval to check activity within.

        Returns:
            bool: True if active, False otherwise. Objects without a time interval are always active.
        """
        if self.time_interval is None:
            return True

        if query_time is not None:
            return query_time in self.time_interval

        if query_interval is not None:
            return self.time_interval.overlaps(query_interval)

        return True
