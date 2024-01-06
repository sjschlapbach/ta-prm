from typing import List, Union, Tuple
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
    """
    A class to represent an environment consisting of a list of custom obstacles.

    ...

    Attributes
    ----------
    obstacles : List[Union[Point, Line, Polygon]]
        A list of obstacles representing the environment.
        Each obstacle is an instance of Point, Line, or Polygon.

    Methods
    -------
    __init__(obstacles: List[Union[Point, Line, Polygon]] = None, filepath: str = None)
        Initialize the Environment object.

    plot(query_time: float = None, fig=None)
        Plots the obstacles in the environment using matplotlib.
        Recurrence parameters are not considered when plotting environments.

    add_obstacles(new_obstacles: List[Union[Point, Line, Polygon]])
        Adds new obstacles to the environment.

    reset()
        Resets the environment by removing all obstacles.

    save(filepath: str)
        Logs the obstacles stored in the environment to a file in JSON format.

    Parameters
    ----------
    obstacles : List[Union[Point, Line, Polygon]], optional
        A list of obstacles to be added to the environment.
        Each obstacle is an instance of Point, Line, or Polygon.
    filepath : str, optional
        The path to the file where the obstacles are stored. If provided, the obstacles will be loaded from the file.

    Raises
    ------
    ValueError
        If an invalid obstacle type is encountered. Only Point, Line, or Polygon are supported.
    """

    def __init__(
        self,
        obstacles: List[Union[Point, Line, Polygon]] = None,
        filepath: str = None,
    ):
        """
        Initialize the Environment object.

        Parameters
        ----------
        obstacles : List[Union[Point, Line, Polygon]], optional
            A list of obstacles to be added to the environment.
            Each obstacle is an instance of Point, Line, or Polygon.
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
        Plots the obstacles in the environment using matplotlib.
        Recurrence parameters are not considered when plotting environments.

        Parameters:
        - query_time (float): The time at which the query is made (optional).
        """
        if fig is None:
            fig = plt.figure(figsize=(8, 8))

        for obstacle in self.obstacles:
            obstacle[1].plot(query_time=query_time, fig=fig)

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
        Logs the obstacles stored in the environment to a file in JSON format.

        Parameters
        ----------
        filepath : str
            The path to the file where the obstacles will be logged.

        Raises
        ------
        ValueError
            If an invalid obstacle type is encountered. Only Point, Line, or Polygon are supported.
        """
        output = {"points": [], "lines": [], "polygons": []}

        for obstacle in self.obstacles:
            json_data = obstacle.export_to_json()

            if isinstance(obstacle, Point):
                output["points"].append(json_data)
            elif isinstance(obstacle, Line):
                output["lines"].append(json_data)
            elif isinstance(obstacle, Polygon):
                output["polygons"].append(json_data)
            else:
                raise ValueError(
                    "Invalid obstacle type. Only Point, Line, or Polygon are supported."
                )

        with open(filepath, "w") as f:
            json.dump(output, f)

    def load(self, filepath: str):
        """
        Loads the obstacles stored in a file into the environment.

        Parameters
        ----------
        filepath : str
            The path to the file where the obstacles are stored.

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
        for pt_obj in obstacles["points"]:
            self.obstacles.append(Point(json_data=pt_obj))

        for ln_obj in obstacles["lines"]:
            self.obstacles.append(Line(json_data=ln_obj))

        for poly_obj in obstacles["polygons"]:
            self.obstacles.append(Polygon(json_data=poly_obj))
