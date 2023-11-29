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
    """

    def __init__(self, radius, interval):
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

    def check_collision(
        self,
        shape: Point | LineString | Polygon,
        query_time: float = None,
        query_interval: Interval = None,
    ):
        """
        Checks if the point is in collision with a given shape. Touching time intervals are considered to be overlapping.

        Args:
            shape (Point, LineString, or Polygon): The shape to check collision with.
            query_time (optional): The specific time to check collision at.
            query_interval (optional): The time interval to check collision within.

        Returns:
            bool: True if collision occurs, False otherwise.
        """
        if isinstance(shape, Point):
            distance = self.geometry.distance(shape)
        elif isinstance(shape, LineString):
            distance = self.geometry.distance(shape)
        elif isinstance(shape, Polygon):
            distance = self.geometry.distance(shape.exterior)
        else:
            raise ValueError(
                "Invalid shape type. Only Point, LineString, or Polygon are supported."
            )

        if query_time is not None:
            if query_time not in self.time_interval:
                return False

        if query_interval is not None:
            if not self.time_interval.overlaps(query_interval):
                return False

        return distance <= self.radius
