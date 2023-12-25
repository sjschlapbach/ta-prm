from shapely.geometry import Point as ShapelyPoint, LineString, Polygon
from shapely import wkt
from pandas import Interval
from matplotlib.patches import Circle
from typing import Union
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
        json_data: dict = None,
    ):
        """
        Initializes a new instance of the Point class.

        Args:
            geometry (ShapelyPoint, optional): The shapely point representing the geometry.
            time_interval (Interval, optional): The pandas interval representing the time interval.
            radius (float, optional): The radius around the point, considered to be in collision.
            json_data (dict, optional): JSON data to load the point from.
        """
        if json_data is not None:
            self.load_from_json(json_data)
            return

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
        shape: Union[ShapelyPoint, LineString, Polygon],
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
        if not self.is_active(query_time, query_interval):
            return

        if fig is None:
            fig = plt.figure()

        plt.figure(fig)

        if self.radius is not None and self.radius > 0:
            poly = self.geometry.buffer(self.radius)
            plt.plot(*poly.exterior.xy, color="black")
        else:
            plt.plot(*self.geometry.xy, color="black", marker="o")

    def export_to_json(self):
        """
        Returns a JSON representation of the point object, using the corresponding exporting
        function of the Geometry class and the stringified version of the geometry.

        Returns:
            str: A JSON representation of the point object.
        """
        if self.geometry is None:
            return {**super().export_to_json(), "geometry": "None"}

        return {**super().export_to_json(), "geometry": self.geometry.wkt}

    def load_from_json(self, json_object):
        """
        Loads the point object from a JSON representation, using the corresponding loading
        function of the Geometry class and the stringified version of the geometry.

        Args:
            json_object (dict): The JSON representation of the point object.

        Returns:
            Object with updated attributes for both the geometry and the parent Geometry object
        """
        super().load_from_json(json_object)

        if json_object["geometry"] == "None":
            self.geometry = None
        else:
            self.geometry = wkt.loads(json_object["geometry"])
