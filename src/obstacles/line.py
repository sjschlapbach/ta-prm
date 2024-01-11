from shapely.geometry import Point, LineString, Polygon
from shapely import wkt
from pandas import Interval
from matplotlib.patches import Circle
from typing import Union
import matplotlib.pyplot as plt
import numpy as np

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

        copy(self) -> 'Line':
            Create a copy of the Line object.

        random(
            min_x: float,
            max_x: float,
            min_y: float,
            max_y: float,
            min_radius: float,
            max_radius: float,
            min_interval: float = 0,
            max_interval: float = 100,
            only_static: bool = False,
            only_dynamic: bool = False,
            random_recurrence: bool = False,
        ):
            Creates a random line.
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
                distance = self.geometry.distance(shape)
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

    def copy(self):
        """
        Create a copy of the line object.

        Returns:
            Line: A copy of the line object.
        """
        return Line(
            geometry=self.geometry,
            time_interval=self.time_interval,
            recurrence=self.recurrence,
            radius=self.radius,
        )

    def random(
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        min_radius: float,
        max_radius: float,
        max_size: float,
        min_interval: float = 0,
        max_interval: float = 100,
        only_static: bool = False,
        only_dynamic: bool = False,
        random_recurrence: bool = False,
    ):
        """
        Creates a random line.

        Args:
            min_x (float, optional): The minimum x-coordinate of the line.
            max_x (float, optional): The maximum x-coordinate of the line.
            min_y (float, optional): The minimum y-coordinate of the line.
            max_y (float, optional): The maximum y-coordinate of the line.
            min_radius (float, optional): The minimum radius of the line.
            max_radius (float, optional): The maximum radius of the line.
            max_size (float, optional): The maximum size of the line.
            min_interval (float, optional): The minimum time interval of the line. Defaults to 0.
            max_interval (float, optional): The maximum time interval of the line. Defaults to 100.
            only_static (bool, optional): Whether to only generate static lines. Defaults to False.
            random_recurrence (bool, optional): Whether to generate a random recurrence. Defaults to False.

        Returns:
            Line: A random line.
        """

        x1 = np.random.uniform(min_x, max_x)
        y1 = np.random.uniform(min_y, max_y)

        # compute range for second point for maximum distance
        size_sqrt = np.sqrt(max_size / 2)
        min_x_rel = max(min_x, x1 - size_sqrt)
        max_x_rel = min(max_x, x1 + size_sqrt)
        min_y_rel = max(min_y, y1 - size_sqrt)
        max_y_rel = min(max_y, y1 + size_sqrt)

        # generate second end point
        x2 = np.random.uniform(min_x_rel, max_x_rel)
        y2 = np.random.uniform(min_y_rel, max_y_rel)

        # generate random radius
        radius = np.random.uniform(min_radius, max_radius)

        # determine if point to be created should be static - 50/50 chance if only_static is False
        if only_static:
            static = True
        elif only_dynamic:
            static = False
        else:
            static = np.random.choice([True, False])

        # if only static lines should be created, do not consider recurrence of time interval
        if static:
            return Line(
                geometry=LineString([(x1, y1), (x2, y2)]),
                radius=radius,
            )

        else:
            # create random time interval
            interval_start = np.random.uniform(min_interval, max_interval)
            interval_end = np.random.uniform(interval_start, max_interval)
            time_interval = Interval(interval_start, interval_end, closed="both")

            # choose random recurrence, if not disabled
            recurrence = (
                Recurrence.random(min_duration=time_interval.length)
                if random_recurrence
                else None
            )

            return Line(
                geometry=LineString([(x1, y1), (x2, y2)]),
                time_interval=time_interval,
                recurrence=recurrence,
                radius=radius,
            )

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
