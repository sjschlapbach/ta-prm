from shapely.geometry import Point, LineString, Polygon
from pandas import Interval
from matplotlib.patches import Circle
from .geometry import Geometry
import matplotlib.pyplot as plt


class Line(Geometry):
    def __init__(
        self,
        geometry: LineString = None,
        time_interval: Interval = None,
        radius: float = 0,
    ):
        """
        Initialize a Line object.

        Args:
            geometry (LineString, optional): The geometry of the line. Defaults to None.
            time_interval (Interval, optional): The time interval during which the line is active. Defaults to None.
            radius (float, optional): The radius of the line. Defaults to 0.
        """
        super().__init__(radius, time_interval)
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
        shape: Point | LineString | Polygon,
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

        if self.is_active(query_time, query_interval):
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
        if fig is None:
            fig = plt.figure()

        plt.figure(fig)
        if query_time is None and query_interval is None or self.time_interval is None:
            plt.plot(*self.geometry.xy, "r-")
            for point in self.geometry.coords:
                circle = Circle(point, self.radius, fill=False)
                plt.gca().add_patch(circle)
        elif query_time is not None and query_time in self.time_interval:
            plt.plot(*self.geometry.xy, "r-")
            for point in self.geometry.coords:
                circle = Circle(point, self.radius, fill=False)
                plt.gca().add_patch(circle)
        elif query_interval is not None and self.time_interval.overlaps(query_interval):
            plt.plot(*self.geometry.xy, "r-")
            for point in self.geometry.coords:
                circle = Circle(point, self.radius, fill=False)
                plt.gca().add_patch(circle)
