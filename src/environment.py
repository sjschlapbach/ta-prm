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

    # TODO - update to new obstacles structure
    # ? idea: solve point, line and polygon obstacles separately to be able to use the correct loading function later on
    def save(self, filepath: str):
        """
        Logs the polygons stored in the environment to a file in JSON format.

        Parameters
        ----------
        filepath : str
            the path to the file where the polygons will be logged
        """
        with open(filepath, "w") as f:
            json.dump([polygon.wkt for polygon in self.polygons], f)

    # TODO - update to new obstacles structure
    def load(self, filepath: str):
        """
        Loads the polygons stored in a file into the environment.

        Parameters
        ----------
        filepath : str
            the path to the file where the polygons are stored
        """
        with open(filepath, "r") as f:
            polygons_wkt = json.load(f)
        self.polygons = [wkt.loads(element) for element in polygons_wkt]
