from shapely.geometry import Point as ShapelyPoint, LineString, Polygon
from shapely import wkt
from pandas import Interval
from matplotlib.patches import Circle
from typing import Union
import matplotlib.pyplot as plt
import numpy as np

from .geometry import Geometry
from src.util.recurrence import Recurrence


class Point(Geometry):
    """
    A class representing a point in geometry.

    Attributes:
        geometry (ShapelyPoint): The shapely point representing the geometry.
        time_interval (Interval, inherited): The pandas interval representing the time interval.
        recurrence (Recurrence, inherited): The recurrence parameter for the point.
        radius (float, inherited): The radius around the point, considered to be in collision.

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

        export_to_json(self):
            Returns a JSON representation of the point object.

        load_from_json(self, json_object):
            Loads the point object from a JSON representation.

        copy(self):
            Creates a copy of the Point object.

        random(
            min_x,
            max_x,
            min_y,
            max_y,
            min_radius=0.1,
            max_radius=10,
            min_interval=0,
            max_interval=100,
            only_static=False,
            only_dynamic=False,
            random_recurrence=False,
        ):
            Generate a random Point object within the specified range.
    """

    def __init__(
        self,
        geometry: ShapelyPoint = None,
        time_interval: Interval = None,
        recurrence: Recurrence = None,
        radius: float = 0,
        json_data: dict = None,
    ):
        """
        Initializes a new instance of the Point class.

        Args:
            geometry (ShapelyPoint, optional): The shapely point representing the geometry.
            time_interval (Interval, optional): The pandas interval representing the time interval.
            recurrence (Recurrence, optional): The recurrence parameter for the point.
            radius (float, optional): The radius around the point, considered to be in collision.
            json_data (dict, optional): JSON data to load the point from.
        """
        if json_data is not None:
            self.load_from_json(json_data)
            return

        super().__init__(radius=radius, interval=time_interval, recurrence=recurrence)
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
        if self.is_active(query_time, query_interval):
            if isinstance(shape, ShapelyPoint):
                distance = self.geometry.distance(shape)
            elif isinstance(shape, LineString):
                distance = self.geometry.distance(shape)
            elif isinstance(shape, Polygon):
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

    def copy(self):
        """
        Returns a copy of the point object.

        Returns:
            Point: A copy of the point object.
        """
        return Point(
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
        min_interval: float = 0,
        max_interval: float = 100,
        only_static: bool = False,
        only_dynamic: bool = False,
        random_recurrence: bool = False,
    ):
        """
        Generate a random Point object within the specified range.

        Args:
            min_x (float): The minimum x-coordinate value.
            max_x (float): The maximum x-coordinate value.
            min_y (float): The minimum y-coordinate value.
            max_y (float): The maximum y-coordinate value.
            min_radius (float, optional): The minimum radius value. Defaults to 0.1.
            max_radius (float, optional): The maximum radius value. Defaults to 10.
            min_interval (float, optional): The minimum interval value. Defaults to 0.
            max_interval (float, optional): The maximum interval value. Defaults to 100.
            only_static (bool, optional): If True, only static points will be created. Defaults to False.
            only_dynamic (bool, optional): If True, only dynamic points will be created. Defaults to False.
            random_recurrence (bool, optional): If True, a random recurrence will be chosen. Defaults to False.

        Returns:
            Point: A randomly generated Point object with the specified parameters.
        """

        x_coord = np.random.uniform(min_x, max_x)
        y_coord = np.random.uniform(min_y, max_y)
        radius = np.random.uniform(min_radius, max_radius)

        # determine if point to be created should be static - 50/50 chance if only_static is False
        if only_static:
            static = True
        elif only_dynamic:
            static = False
        else:
            static = np.random.choice([True, False])

        # if only static points should be created, do not consider recurrence of time interval
        if static:
            return Point(
                geometry=ShapelyPoint(x_coord, y_coord),
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

            return Point(
                geometry=ShapelyPoint(x_coord, y_coord),
                time_interval=time_interval,
                recurrence=recurrence,
                radius=radius,
            )

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
