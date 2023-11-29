from shapely.geometry import Point as ShapelyPoint, LineString, Polygon
from pandas import Interval
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from .geometry import Geometry


class Point(Geometry):
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

        check_collision(self, shape, query_time=None, query_interval=None):
            Checks if the point is in collision with a given shape. Touching time intervals are considered to be overlapping.

        plot(self, query_time=None, query_interval=None, fig=None):
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
        super().__init__(radius, time_interval)
        self.geometry = geometry

    def set_geometry(self, x: float, y: float):
        """
        Sets the geometry from a coordinate pair.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.
        """
        self.geometry = ShapelyPoint(x, y)

    def check_collision(
        self,
        shape: ShapelyPoint | LineString | Polygon,
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
            bool: True if collision occurs, False otherwise. Objects without a time interval are considered to be always active.
        """
        if isinstance(shape, ShapelyPoint):
            distance = self.geometry.distance(shape)
        elif isinstance(shape, LineString):
            distance = self.geometry.distance(shape)
        elif isinstance(shape, Polygon):
            distance = self.geometry.distance(shape.exterior)
        else:
            raise ValueError(
                "Invalid shape type. Only Point, LineString, or Polygon are supported."
            )

        if self.is_active(query_time, query_interval):
            return distance <= self.radius
        else:
            return False

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
