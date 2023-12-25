from typing import List, Union
from shapely.geometry import (
    Polygon as ShapelyPolygon,
    Point as ShapelyPoint,
    LineString as ShapelyLine,
)
from shapely import wkt
import json
import matplotlib.pyplot as plt

from src.obstacles.point import Point
from src.obstacles.line import Line
from src.obstacles.polygon import Polygon


class Environment:
    # TODO - udpate docstring after methods have been updated
    """
    A class to represent an environment consisting of a list of shapely polygon objects.

    ...

    Attributes
    ----------
    polygons : List[Polygon]
        a list of shapely polygon objects representing the environment

    Methods
    -------
    plot()
        Plots the polygons in the environment using matplotlib.

    closest_polygon_distance(point: Point) -> float
        Computes the distance between a shapely point object and the closest polygon in the environment.

    closest_line_distance(line: LineString) -> float
        Computes the distance between a shapely line object and the closest polygon in the environment.

    change_polygons(new_polygons: List[Polygon])
        Changes the polygons stored in the environment.

    save(filepath: str)
        Logs the polygons stored in the environment to a file in JSON format.

    load(filepath: str)
        Loads the polygons stored in a file into the environment.
    """

    def __init__(
        self, obstacles: List[Union[Point, Line, Polygon]] = None, filepath: str = None
    ):
        """
        Initialize the Environment object.

        Parameters
        ----------
        obstacles : List[Union[Point, Line, Polygon]], optional
            A list of obstacles to be added to the environment.
        filepath : str, optional
            The path to the file where the obstacles are stored. If provided, the obstacles will be loaded from the file.
        """
        self.obstacles = []
        if filepath is not None:
            self.load(filepath)

        if obstacles is not None:
            self.obstacles += obstacles

    def plot(self, query_time: float = None, fig=None):
        """
        Plots the polygons in the environment using matplotlib.

        Parameters:
        - query_time (float): The time at which the query is made (optional).
        """
        if fig is None:
            fig = plt.figure(figsize=(8, 8))

        for obstacle in self.obstacles:
            obstacle.plot(query_time=query_time, fig=fig)

    def add_obstacles(self, new_obstacles: List[Union[Point, Line, Polygon]]):
        """
        Adds new obstacles to the environment.

        Parameters
        ----------
        new_obstacles : List[Union[Point, Line, Polygon]]
            A list of obstacles to be added to the environment.
            Each obstacle can be a Point, Line, or Polygon object.
        """
        self.obstacles += new_obstacles

    def reset(self):
        """
        Resets the environment by removing all obstacles.
        """
        self.obstacles = []

    def save(self, filepath: str):
        """
        Logs the polygons stored in the environment to a file in JSON format.

        Parameters
        ----------
        filepath : str
            The path to the file where the polygons will be logged.

        Raises
        ------
        ValueError
            If an invalid obstacle type is encountered. Only Point, Line, or Polygon are supported.
        """
        output = {"points": [], "lines": [], "polygons": []}

        for obstacle in self.obstacles:
            if isinstance(obstacle, Point):
                output["points"].append(obstacle.export_to_json())
            elif isinstance(obstacle, Line):
                output["lines"].append(obstacle.export_to_json())
            elif isinstance(obstacle, Polygon):
                output["polygons"].append(obstacle.export_to_json())
            else:
                raise ValueError(
                    "Invalid obstacle type. Only Point, Line, or Polygon are supported."
                )

        with open(filepath, "w") as f:
            json.dump(output, f)

    def load(self, filepath: str):
        """
        Loads the polygons stored in a file into the environment.

        Parameters
        ----------
        filepath : str
            The path to the file where the polygons are stored.

        Raises
        ------
        FileNotFoundError
            If the file specified by `filepath` does not exist.

        JSONDecodeError
            If there is an error decoding the JSON file.

        Returns
        -------
        None
        """
        obstacles = {}

        # load obstacles from file
        with open(filepath, "r") as f:
            obstacles = json.load(f)

        # convert obstacles to custom class objects and store them in class variable
        for pt in obstacles["points"]:
            point = Point()
            point.load_from_json(pt)
            self.obstacles.append(point)

        for ln in obstacles["lines"]:
            line = Line()
            line.load_from_json(ln)
            self.obstacles.append(line)

        for poly in obstacles["polygons"]:
            polygon = Polygon()
            polygon.load_from_json(poly)
            self.obstacles.append(polygon)
