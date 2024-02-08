from shapely.geometry import Point, LineString, Polygon as ShapelyPolygon
from shapely import wkt
from pandas import Interval
from matplotlib.patches import Circle
from typing import Union
import matplotlib.pyplot as plt
import numpy as np

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
        __init__(self, geometry=None, time_interval=None, radius=0, json_data=None):
            Initializes a new instance of the Polygon class.

        set_geometry(self, points):
            Sets the geometry from a list of coordinate pairs.

        check_collision(self, shape, query_time=None, query_interval=None):
            Checks if the polygon is in collision with a given shape. Touching time intervals are considered to be overlapping.

        plot(self, query_time=None, query_interval=None, color="black", fill_color=None, opactiy=1, show_inactive=False, inactive_color="grey", inactive_fill_color=None, fig=None):
            Plots the polygon with a circle of the corresponding radius around it.
            Optionally, only shows the polygon with the circle if it is active.

        copy(self):
            Creates a copy of the Polygon object.

        save_to_json(self, filepath):
            Saves the polygon to a JSON file.

        load_from_json(self, json_data):
            Loads the polygon from JSON data.
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

    def plot(
        self,
        query_time: float = None,
        query_interval: Interval = None,
        color: str = "black",
        fill_color: str = None,
        opactiy: float = 1,
        show_inactive: bool = False,
        inactive_color: str = "grey",
        inactive_fill_color: str = None,
        fig=None,
    ):
        """
        Plots the polygon with a circle of the corresponding radius around it.
        Optionally, only shows the polygon with the circle if it is active.

        Args:
            query_time (optional): The specific time to check if the polygon is active.
            query_interval (optional): The time interval to check if the polygon is active.
            fig (optional): The figure to plot on. If not provided, a new figure will be created.
            fill_color (optional): The color to fill the polygon with.
            opactiy (optional): The opacity of the polygon.
            show_inactive (optional): Whether to show the polygon if it is inactive.
            inactive_color (optional): The color to use for inactive polygons.
            inactive_fill_color (optional): The fill color to use for inactive polygons.
            fig (optional): The figure to plot on. If not provided, a new figure will be created.
        """
        if not self.is_active(query_time, query_interval):
            if show_inactive:
                if self.radius is not None and self.radius > 0:
                    poly = self.geometry.buffer(self.radius)
                    plt.plot(*poly.exterior.xy, color=inactive_color)
                    plt.fill(*poly.exterior.xy, color=inactive_fill_color, alpha=0.05)
                else:
                    plt.plot(*self.geometry.exterior.xy, color=inactive_color)
                    plt.fill(
                        *self.geometry.exterior.xy,
                        color=inactive_fill_color,
                        alpha=0.05
                    )
            return

        if fig is None:
            fig = plt.figure()

        plt.figure(fig)

        if self.radius is not None and self.radius > 0:
            poly = self.geometry.buffer(self.radius)
            plt.plot(*poly.exterior.xy, color=color)
            plt.fill(*poly.exterior.xy, color=fill_color, alpha=opactiy)
        else:
            plt.plot(*self.geometry.exterior.xy, color=color)
            plt.fill(*self.geometry.exterior.xy, color=fill_color, alpha=opactiy)

    def copy(self):
        """
        Creates a copy of the polygon object.

        Returns:
            Polygon: A copy of the polygon object.
        """
        return Polygon(
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
        max_size: float,
        min_radius: float,
        max_radius: float,
        min_points: int = 3,
        max_points: int = 7,
        min_interval: float = 0,
        max_interval: float = 100,
        only_static: bool = False,
        only_dynamic: bool = False,
        random_recurrence: bool = False,
    ):
        """
        Creates a random polygon.

        Args:
            min_points (int): The minimum number of points on the polygon edge.
            max_points (int): The maximum number of points on the polygon edge.
            min_x (float): The minimum x coordinate.
            max_x (float): The maximum x coordinate.
            min_y (float): The minimum y coordinate.
            max_y (float): The maximum y coordinate.
            max_size (float): The maximum size of the polygon (without radius).
            min_radius (float): The minimum radius of the polygon.
            max_radius (float): The maximum radius of the polygon.
            min_interval (float, optional): The minimum time interval.
            max_interval (float, optional): The maximum time interval.
            only_static (bool, optional): Whether to generate only static polygons.
            only_dynamic (bool, optional): Whether to generate only dynamic polygons.
            random_recurrence (bool, optional): Whether to generate a random recurrence.

        Returns:
            Polygon: A random polygon.
        """

        # generate a random number of points
        num_points = np.random.randint(min_points, max_points)

        # generate first point
        pt1 = (np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))
        points = [pt1]

        # generate the rest of the points at a maximum distance of max_size / 2 from pt1
        size_sqrt = np.sqrt(max_size / 2)
        min_x_rel = max(pt1[0] - size_sqrt, min_x)
        max_x_rel = min(pt1[0] + size_sqrt, max_x)
        min_y_rel = max(pt1[1] - size_sqrt, min_y)
        max_y_rel = min(pt1[1] + size_sqrt, max_y)

        for _ in range(num_points - 1):
            pt = (
                np.random.uniform(min_x_rel, max_x_rel),
                np.random.uniform(min_y_rel, max_y_rel),
            )
            points.append(pt)

        # create the polygon
        poly = ShapelyPolygon(points)
        poly = ShapelyPolygon(poly.convex_hull)

        # generate a random radius
        radius = np.random.uniform(min_radius, max_radius)

        # determine if polygon to be created should be static - 50/50 chance if only_static is False
        if only_static:
            static = True
        elif only_dynamic:
            static = False
        else:
            static = np.random.choice([True, False])

        # if only static lines should be created, do not consider recurrence of time interval
        if static:
            return Polygon(
                geometry=poly,
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

            return Polygon(
                geometry=poly,
                time_interval=time_interval,
                recurrence=recurrence,
                radius=radius,
            )

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
