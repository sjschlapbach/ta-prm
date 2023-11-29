from enum import Enum
from shapely.geometry import Point as ShapelyPoint, LineString, Polygon
from pandas import Interval
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

# TODO: implement geometry wrapper which contains these enums
# ! on creation of edges / a scenario, point objects with the same geometry are created multiple times
# class Recurrence(Enum):
#     NONE = "none"
#     HOURLY = "hourly"
#     DAILY = "daily"
#     YEARLY = "yearly"


class Point:
    """
    A class representing a point in geometry.

    Attributes:
        geometry (ShapelyPoint): The shapely point representing the geometry.
        time_interval (Interval): The pandas interval representing the time interval.
        radius (float): The radius around the point, considered to be in collision.

    Methods:
        __init__(self, geometry=None, time_interval=None, radius=0):
            Initializes a new instance of the Point class.

        set_geometry(self, x, y):
            Sets the geometry from a coordinate pair.

        set_interval(self, lower_bound, upper_bound):
            Sets the time interval from a lower and upper bound.

        set_radius(self, radius):
            Sets the radius around the point, considered to be in collision.

        check_collision(self, shape, query_time=None, query_interval=None):
            Checks if the point is in collision with a given shape.

        plot(self, query_time=None, query_interval=None):
            Plots the point with a circle of the corresponding radius around it.
            Optionally, only shows the point with the circle if it is active.
    """

    def __init__(
        self,
        geometry: ShapelyPoint = None,
        time_interval: Interval = None,
        radius: float = 0,
    ):
        """
        Initializes a new instance of the Point class.

        Args:
            geometry (ShapelyPoint): The shapely point representing the geometry.
            time_interval (Interval): The pandas interval representing the time interval.
            radius (float): The radius around the point, considered to be in collision.
        """
        self.geometry = geometry
        self.time_interval = time_interval
        self.radius = radius

    def set_geometry(self, x: float, y: float):
        """
        Sets the geometry from a coordinate pair.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.
        """
        self.geometry = ShapelyPoint(x, y)

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
        shape: ShapelyPoint | LineString | Polygon,
        query_time: float = None,
        query_interval: Interval = None,
    ):
        """
        Checks if the point is in collision with a given shape. Touching time intervals are considered to be overlapping.

        Args:
            shape (ShapelyPoint, LineString, or Polygon): The shape to check collision with.
            query_time (optional): The specific time to check collision at.
            query_interval (optional): The time interval to check collision within.

        Returns:
            bool: True if collision occurs, False otherwise.
        """
        if isinstance(shape, ShapelyPoint):
            distance = self.geometry.distance(shape)
        elif isinstance(shape, LineString):
            distance = self.geometry.distance(shape)
        elif isinstance(shape, Polygon):
            distance = self.geometry.distance(shape.exterior)
        else:
            raise ValueError(
                "Invalid shape type. Only ShapelyPoint, LineString, or Polygon are supported."
            )

        if query_time is not None:
            if query_time not in self.time_interval:
                return False

        if query_interval is not None:
            if not self.time_interval.overlaps(query_interval):
                return False

        return distance <= self.radius

    def plot(self, query_time: float = None, query_interval: Interval = None, fig=None):
        """
        Plots the point with a circle of the corresponding radius around it.
        Optionally, only shows the point with the circle if it is active.

        Args:
            query_time (optional): The specific time to check if the point is active.
            query_interval (optional): The time interval to check if the point is active.
            fig (optional): The figure to plot on. If not provided, a new figure will be created.
        """
        if fig is None:
            fig = plt.figure()

        # Plot the point and circle around it
        plt.figure(fig)
        if query_time is None and query_interval is None or self.time_interval is None:
            plt.plot(self.geometry.x, self.geometry.y, "ro")
            circle = Circle((self.geometry.x, self.geometry.y), self.radius, fill=False)
            plt.gca().add_patch(circle)
        elif query_time is not None and query_time in self.time_interval:
            plt.plot(self.geometry.x, self.geometry.y, "ro")
            circle = Circle((self.geometry.x, self.geometry.y), self.radius, fill=False)
            plt.gca().add_patch(circle)
        elif query_interval is not None and self.time_interval.overlaps(query_interval):
            plt.plot(self.geometry.x, self.geometry.y, "ro")
            circle = Circle((self.geometry.x, self.geometry.y), self.radius, fill=False)
            plt.gca().add_patch(circle)
