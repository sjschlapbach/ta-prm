from shapely.geometry import Point, LineString, Polygon as ShapelyPolygon
from shapely import wkt
from pandas import Interval
from matplotlib.patches import Circle
from typing import Union
import matplotlib.pyplot as plt

from .geometry import Geometry
from src.util.recurrence import Recurrence


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
        recurrence: Recurrence = None,
        radius: float = 0,
        json_data: dict = None,
    ):
        """
        Initializes a new instance of the Polygon class.

        Args:
            geometry (Polygon, optional): The shapely polygon representing the geometry.
            time_interval (Interval, optional): The pandas interval representing the time interval.
            recurrence (Recurrence, optional): The recurrence parameter for the polygon.
            radius (float, optional): The radius around the polygon, considered to be in collision.
            json_data (dict, optional): Optional JSON data to load the polygon from.
        """
        if json_data is not None:
            self.load_from_json(json_data)
            return

        super().__init__(radius=radius, interval=time_interval, recurrence=recurrence)
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
        if self.is_active(query_time, query_interval):
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

    def export_to_json(self):
        """
        Returns a JSON representation of the polygon object, using the corresponding exporting
        function of the Geometry class and the stringified version of the geometry.

        Returns:
            str: A JSON representation of the polygon object.
        """
        if self.geometry is None:
            return {**super().export_to_json(), "geometry": "None"}

        return {**super().export_to_json(), "geometry": self.geometry.wkt}

    def load_from_json(self, json_object):
        """
        Loads the polygon object from a JSON representation, using the corresponding loading
        function of the Geometry class and the stringified version of the geometry.

        Args:
            json_object (dict): The JSON representation of the polygon object.

        Returns:
            Object with updated attributes for both the geometry and the parent Geometry object
        """
        super().load_from_json(json_object)

        if json_object["geometry"] == "None":
            self.geometry = None
        else:
            self.geometry = wkt.loads(json_object["geometry"])
