from shapely.geometry import Point, LineString, Polygon as ShapelyPolygon
from pandas import Interval
from matplotlib.patches import Circle
from typing import Union
import matplotlib.pyplot as plt

from .geometry import Geometry


class Polygon(Geometry):
    """
    A class representing a polygon in geometry.

    Attributes:
        geometry (Polygon): The shapely polygon representing the geometry.
        time_interval (Interval): The pandas interval representing the time interval.
        radius (float): The radius around the polygon, considered to be in collision.

    Methods:
        __init__(self, geometry=None, time_interval=None, radius=0):
            Initializes a new instance of the Polygon class.

        set_geometry(self, points):
            Sets the geometry from a list of coordinate pairs.

        check_collision(self, shape, query_time=None, query_interval=None):
            Checks if the polygon is in collision with a given shape. Touching time intervals are considered to be overlapping.

        plot(self, query_time=None, query_interval=None, fig=None):
            Plots the polygon with a circle of the corresponding radius around it.
            Optionally, only shows the polygon with the circle if it is active.
    """

    def __init__(
        self,
        geometry: ShapelyPolygon = None,
        time_interval: Interval = None,
        radius: float = 0,
    ):
        """
        Initializes a new instance of the Polygon class.

        Args:
            geometry (Polygon): The shapely polygon representing the geometry.
            time_interval (Interval): The pandas interval representing the time interval.
            radius (float): The radius around the polygon, considered to be in collision.
        """
        super().__init__(radius, time_interval)
        self.geometry = geometry

    def set_geometry(self, points: list[tuple[float, float]]):
        """
        Sets the geometry from a list of coordinate pairs.

        Args:
            points (list): A list of coordinate pairs [(x1, y1), (x2, y2), ...].
        """
        self.geometry = ShapelyPolygon(points)

    def check_collision(
        self,
        shape: Union[Point, LineString, ShapelyPolygon],
        query_time: float = None,
        query_interval: Interval = None,
    ):
        """
        Checks if the polygon is in collision with a given shape. Touching time intervals are considered to be overlapping.

        Args:
            shape (Point, LineString, or Polygon): The shape to check collision with.
            query_time (optional): The specific time to check collision at.
            query_interval (optional): The time interval to check collision within.

        Returns:
            bool: True if collision occurs, False otherwise. Objects without a time interval are considered to be always active.
        """
        if isinstance(shape, Point):
            distance = self.geometry.distance(shape)
        elif isinstance(shape, LineString):
            distance = self.geometry.distance(shape)
        elif isinstance(shape, ShapelyPolygon):
            distance = self.geometry.distance(shape)
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
        Plots the polygon with a circle of the corresponding radius around it.
        Optionally, only shows the polygon with the circle if it is active.

        Args:
            query_time (optional): The specific time to check if the polygon is active.
            query_interval (optional): The time interval to check if the polygon is active.
            fig (optional): The figure to plot on. If not provided, a new figure will be created.
        """
        if not self.is_active(query_time, query_interval):
            return

        if fig is None:
            fig = plt.figure()

        plt.figure(fig)

        if self.radius is not None and self.radius > 0:
            poly = self.geometry.buffer(self.radius)
            plt.plot(*poly.exterior.xy, color="red")
        else:
            plt.plot(*self.geometry.exterior.xy, color="red")
