from typing import List, Union
from shapely.geometry import (
    Polygon as ShapelyPolygon,
    Point as ShapelyPoint,
    LineString as ShapelyLine,
)
from shapely import wkt
import json
import matplotlib.pyplot as plt
import numpy as np

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

    add_obstacles(new_obstacles: List[Union[Point, Line, Polygon]])
        Adds new obstacles to the environment.

    reset()
        Resets the environment by removing all obstacles.

    save(filepath: str)
        Logs the obstacles stored in the environment to a file in JSON format.

    load(filepath: str)
        Loads the obstacles stored in a file into the environment.

    Parameters
    ----------
    obstacles : List[Union[Point, Line, Polygon]], optional
        A list of obstacles to be added to the environment.
        Each obstacle is an instance of Point, Line, or Polygon.

    filepath : str, optional
        The path to the file where the obstacles are stored.
        If provided, the obstacles will be loaded from the file.

    Raises
    ------
    RuntimeError
        If an invalid obstacle type is encountered. Only Point, Line, or Polygon are supported.

    FileNotFoundError
        If the file specified by `filepath` does not exist.

    JSONDecodeError
        If there is an error decoding the JSON file.
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
            The path to the file where the obstacles are stored.
            If provided, the obstacles will be loaded from the file.
        """
        self.obstacles = []

        if filepath is not None:
            self.load(filepath)

        if obstacles is not None:
            self.obstacles += obstacles

    def plot(self, query_time: float = None, fig=None):
        """
        Plots the obstacles in the environment using matplotlib.

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

    def add_random_obstacles(
        self,
        num_points: int,
        num_lines: int,
        num_polygons: int,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        min_poly_points: int = 3,
        max_poly_points: int = 7,
        max_size: float = 5,
        min_radius: float = 0.1,
        max_radius: float = 10,
        min_interval: float = 0,
        max_interval: float = 100,
        only_static: bool = False,
        only_dynamic: bool = False,
        random_recurrence: bool = False,
        seed: int = None,
    ):
        """
        Adds random obstacles to the environment.

        Parameters
        ----------
        num_points : int, optional
            The number of random points to be added to the environment.

        num_lines : int, optional
            The number of random lines to be added to the environment.

        num_polygons : int, optional
            The number of random polygons to be added to the environment.

        min_x : float, optional
            The minimum x-coordinate of the environment.

        max_x : float, optional
            The maximum x-coordinate of the environment.

        min_y : float, optional
            The minimum y-coordinate of the environment.

        max_y : float, optional
            The maximum y-coordinate of the environment.

        min_poly_points : int, optional
            The minimum number of points on the random polygons.

        max_poly_points : int, optional
            The maximum number of points on the random polygons.

        max_size : float, optional
            The maximum size of the random polygons (without radius).

        min_radius : float, optional
            The minimum radius of the random circles.

        max_radius : float, optional
            The maximum radius of the random circles.

        min_interval : float, optional
            The earliest start time of a random obstacle.

        max_interval : float, optional
            The latest end time of a random obstacle.

        only_static : bool, optional
            Whether to only generate static obstacles.

        only_dynamic : bool, optional
            Whether to only generate dynamic obstacles.

        random_recurrence : bool, optional
            Whether to generate a random recurrence.

        seed : int, optional
            The seed to use for the random number generator.
        """
        # set seed
        if seed is not None:
            np.random.seed(seed)

        # add random points
        for _ in range(num_points):
            point = Point.random(
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
                min_radius=min_radius,
                max_radius=max_radius,
                min_interval=min_interval,
                max_interval=max_interval,
                only_static=only_static,
                only_dynamic=only_dynamic,
                random_recurrence=random_recurrence,
            )
            self.obstacles.append(point)

        # add random lines
        for _ in range(num_lines):
            line = Line.random(
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
                min_radius=min_radius,
                max_radius=max_radius,
                min_interval=min_interval,
                max_interval=max_interval,
                max_size=max_size,
                only_static=only_static,
                only_dynamic=only_dynamic,
                random_recurrence=random_recurrence,
            )
            self.obstacles.append(line)

        # add random polygons
        for _ in range(num_polygons):
            polygon = Polygon.random(
                min_points=min_poly_points,
                max_points=max_poly_points,
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
                max_size=max_size,
                min_radius=min_radius,
                max_radius=max_radius,
                min_interval=min_interval,
                max_interval=max_interval,
                only_static=only_static,
                only_dynamic=only_dynamic,
                random_recurrence=random_recurrence,
            )
            self.obstacles.append(polygon)

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
        RuntimeError
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
                raise RuntimeError(
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
