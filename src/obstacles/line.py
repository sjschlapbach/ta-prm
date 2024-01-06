from shapely.geometry import Point, LineString, Polygon
from shapely import wkt
from pandas import Interval
from matplotlib.patches import Circle
from typing import Union
import matplotlib.pyplot as plt

from .geometry import Geometry
from src.util.recurrence import Recurrence


class Line(Geometry):
    """
    A class representing a line in geometry.

    Attributes:
        geometry (LineString): The geometry of the line.
        time_interval (Interval): The time interval during which the line is active.
        recurrence (Recurrence): The recurrence frequency of the line.
        radius (float): The radius of the line.

    Methods:
        __init__(self, geometry: LineString = None, time_interval: Interval = None, recurrence: Recurrence = None, radius: float = 0, json_data: dict = None):
            Initialize a Line object.

        set_geometry(self, points: list[tuple[float, float]]):
            Set the geometry of the line.

        check_collision(self, shape: Union[Point, LineString, Polygon], query_time: float = None, query_interval: Interval = None) -> bool:
            Check if the line collides with a given shape.

        plot(self, query_time: float = None, query_interval: Interval = None, fig=None):
            Plot the line.

        export_to_json(self) -> dict:
            Returns a JSON representation of the line object.

        load_from_json(self, json_data: dict):
            Loads the line object from a JSON representation.
    """

    def __init__(
        self,
        geometry: LineString = None,
        time_interval: Interval = None,
        recurrence: Recurrence = None,
        radius: float = 0,
        json_data: dict = None,
    ):
        """
        Initialize a Line object.

        Args:
            geometry (LineString, optional): The geometry of the line. Defaults to None.
            time_interval (Interval, optional): The time interval during which the line is active. Defaults to None.
            recurrence (Recurrence, optional): The recurrence frequency of the line. Defaults to None.
            radius (float, optional): The radius of the line. Defaults to 0.
            json_data (dict, optional): JSON data to initialize the line object. If provided, other arguments will be ignored. Defaults to None.
        """
        if json_data is not None:
            self.load_from_json(json_data)
            return

        super().__init__(radius=radius, interval=time_interval, recurrence=recurrence)
        self.geometry = geometry

    def set_geometry(self, points: list[tuple[float, float]]):
        """
        Set the geometry of the line.

        Args:
            points (list[tuple[float, float]]): The list of points defining the line.
        """
        self.geometry = LineString(points)

    def check_collision(
        self,
        shape: Union[Point, LineString, Polygon],
        query_time: float = None,
        query_interval: Interval = None,
    ) -> bool:
        """
        Check if the line collides with a given shape.

        Args:
            shape (LineString | Polygon): The shape to check collision with.
            query_time (float, optional): The query time. Defaults to None.
            query_interval (Interval, optional): The query time interval. Defaults to None.

        Returns:
            bool: True if collision occurs, False otherwise.

        Raises:
            ValueError: If the shape type is not supported.
        """
        if self.is_active(query_time, query_interval):
            if isinstance(shape, Point):
                distance = self.geometry.distance(shape)
            elif isinstance(shape, LineString):
                distance = self.geometry.distance(shape)
            elif isinstance(shape, Polygon):
                distance = self.geometry.distance(shape.exterior)
            else:
                raise ValueError(
                    "Invalid shape type. Only LineString or Polygon are supported."
                )

            return distance <= self.radius
        else:
            return False

    def plot(self, query_time: float = None, query_interval: Interval = None, fig=None):
        """
        Plot the line.

        Args:
            query_time (float, optional): The query time. Defaults to None.
            query_interval (Interval, optional): The query time interval. Defaults to None.
            fig: The figure to plot on. Defaults to None.
        """
        if not self.is_active(query_time, query_interval):
            return

        if fig is None:
            fig = plt.figure()

        plt.figure(fig)

        if self.radius is not None and self.radius > 0:
            poly = self.geometry.buffer(self.radius)
            plt.plot(*poly.exterior.xy, color="blue")
        else:
            plt.plot(*self.geometry.xy, color="blue")

    def export_to_json(self):
        """
        Returns a JSON representation of the line object, using the corresponding exporting
        function of the Geometry class and the stringified version of the geometry.

        Returns:
            str: A JSON representation of the line object.
        """
        if self.geometry is None:
            return {**super().export_to_json(), "geometry": "None"}

        return {**super().export_to_json(), "geometry": self.geometry.wkt}

    def load_from_json(self, json_object):
        """
        Loads the line object from a JSON representation, using the corresponding loading
        function of the Geometry class and the stringified version of the geometry.

        Args:
            json_object (dict): The JSON representation of the line object.

        Returns:
            Object with updated attributes for both the geometry and the parent Geometry object
        """
        super().load_from_json(json_object)

        if json_object["geometry"] == "None":
            self.geometry = None
        else:
            self.geometry = wkt.loads(json_object["geometry"])
