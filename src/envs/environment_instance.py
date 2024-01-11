from typing import List, Union, Dict, Tuple
from pandas import Interval
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

from shapely.geometry import (
    Polygon as ShapelyPolygon,
    Point as ShapelyPoint,
    LineString as ShapelyLine,
)

from .environment import Environment
from src.obstacles.point import Point
from src.obstacles.line import Line
from src.obstacles.polygon import Polygon
from src.util.recurrence import Recurrence


class EnvironmentInstance:
    """
    Represents an instance of the environment.

    Attributes:
        query_interval (Interval): The time interval for which the obstacles are active.
        dim_x (List[int]): The range of x-coordinates for the scenario.
        dim_y (List[int]): The range of y-coordinates for the scenario.
        static_obstacles (Dict[int, Union[Point, Line, Polygon]]): Dictionary of static obstacles.
        dynamic_obstacles (Dict[int, Union[Point, Line, Polygon]]): Dictionary of dynamic obstacles.
        resolution (int): The resolution used for computing the spatial indices.
        static_idx (List[List[List[int]]]): Spatial indices for static obstacles.
        dynamic_idx (List[List[List[int]]]): Spatial indices for dynamic obstacles.
        spacing_x (float): Spacing in the x-direction.
        spacing_y (float): Spacing in the y-direction.

    Methods:
        __init__(self, environment: Environment, query_interval: Interval, scenario_range_x: Tuple[int, int], scenario_range_y: Tuple[int, int], resolution: int = 20)
        compute_spatial_indices(self, resolution: int) -> Tuple[List[List[List[int]]], List[List[List[int]]], float, float]
    """

    def __init__(
        self,
        environment: Environment,
        query_interval: Interval,
        scenario_range_x: Tuple[int, int],
        scenario_range_y: Tuple[int, int],
        resolution: int = 20,
    ):
        """
        Initialize an instance of the EnvironmentInstance class.

        Args:
            environment (Environment): The environment containing the obstacles.
            query_interval (Interval): The time interval for which the obstacles are active.
            scenario_range_x (Tuple[int, int]): The range of x-coordinates for the scenario.
            scenario_range_y (Tuple[int, int]): The range of y-coordinates for the scenario.
            resolution (int, optional): The resolution used for computing the spatial indices. Defaults to 20.
        """
        # set variables from environment
        self.query_interval = query_interval
        self.dim_x = [scenario_range_x[0], scenario_range_x[1]]
        self.dim_y = [scenario_range_y[0], scenario_range_y[1]]

        # initialize obstacle lists and counter for indexing
        self.static_obstacles: Dict[int, Union[Point, Line, Polygon]] = {}
        self.dynamic_obstacles: Dict[int, Union[Point, Line, Polygon]] = {}
        counter = 1

        print("Loading obstacles into environment instance...")
        for obstacle in tqdm(environment.obstacles):
            # if dimensions are specified for the environment, check that the obstacle is contained
            env_poly = ShapelyPolygon(
                [
                    (scenario_range_x[0], scenario_range_y[0]),
                    (scenario_range_x[0], scenario_range_y[1]),
                    (scenario_range_x[1], scenario_range_y[1]),
                    (scenario_range_x[1], scenario_range_y[0]),
                ]
            )
            if not obstacle.check_collision(shape=env_poly):
                continue

            # if the obstacle is static, add it to the static obstacles
            if obstacle.time_interval is None:
                obs_copy = obstacle.copy()
                obs_copy.recurrence = Recurrence.NONE
                self.static_obstacles[counter] = obs_copy
                counter += 1
                continue

            # check if the obstacle is active during the query interval
            if obstacle.is_active(query_interval=query_interval):
                if obstacle.recurrence == Recurrence.NONE:
                    # if entire query interval is covered, add it to the static obstacles
                    if (
                        obstacle.time_interval.left <= query_interval.left
                        and obstacle.time_interval.right >= query_interval.right
                    ):
                        obs_copy = obstacle.copy()
                        obs_copy.time_interval = None
                        self.static_obstacles[counter] = obs_copy
                        counter += 1
                        continue

                    # otherwise add it to the dynamic obstacles
                    else:
                        obs_copy = obstacle.copy()
                        self.dynamic_obstacles[counter] = obs_copy
                        counter += 1
                        continue

                else:
                    # check if the obtacle occurence overlapping with the query interval covers it entirely
                    obstacle_start = obstacle.time_interval.left
                    obstacle_end = obstacle.time_interval.right
                    query_start = query_interval.left
                    query_end = query_interval.right

                    # select the correct occurence overlapping with the query interval
                    delta = query_start - obstacle_start
                    rec_length = obstacle.recurrence.get_seconds()
                    occurence = delta // rec_length

                    # compute the start and end of the overlapping occurence
                    occurence_start = obstacle_start + occurence * rec_length
                    occurence_end = occurence_start + obstacle.time_interval.length

                    # if the occurence covers the entire query interval, remove all unnecessary information and
                    # add it to the static obstacles
                    if (
                        delta >= 0
                        and occurence_start <= query_start
                        and occurence_end >= query_end
                    ):
                        obs_copy = obstacle.copy()
                        obs_copy.time_interval = None
                        obs_copy.recurrence = Recurrence.NONE
                        self.static_obstacles[counter] = obs_copy
                        counter += 1
                        continue

                    # otherwise, add it to the dynamic obstacles
                    else:
                        obs_copy = obstacle.copy()
                        self.dynamic_obstacles[counter] = obs_copy
                        counter += 1
                        continue

                raise ValueError(
                    "Obstacle was not added to any obstacle list, despite being active"
                )

            else:
                continue

        print("Creating spatial index for environment instance...")
        self.resolution = resolution
        (
            self.static_idx,
            self.dynamic_idx,
            self.spacing_x,
            self.spacing_y,
        ) = self.compute_spatial_indices(resolution)

    def compute_spatial_indices(self, resolution: int):
        """
        Compute spatial indices for static and dynamic obstacles in the environment.

        Args:
            resolution (int): The number of cells in each dimension of the grid.

        Returns:
            tuple: A tuple containing the static obstacle indices, dynamic obstacle indices,
                   spacing in the x-direction, and spacing in the y-direction.

        """
        static_arr = [[[]] * resolution for i in range(resolution)]
        dynamic_arr = [[[]] * resolution for i in range(resolution)]
        spacing_x = (self.dim_x[1] - self.dim_x[0]) / resolution
        spacing_y = (self.dim_y[1] - self.dim_y[0]) / resolution

        for kx in range(resolution):
            for ky in range(resolution):
                # create a polygon for the current cell
                x1 = self.dim_x[0] + kx * spacing_x
                x2 = self.dim_x[0] + (kx + 1) * spacing_x
                y1 = self.dim_y[0] + ky * spacing_y
                y2 = self.dim_y[0] + (ky + 1) * spacing_y
                cell_poly = ShapelyPolygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])

                # initialize static and dynamic obstacle lists for the current cell
                static_ids = []
                dynamic_ids = []

                # find all static obstacles in the current cell
                for static_idx in self.static_obstacles:
                    obstacle = self.static_obstacles[static_idx]
                    if obstacle.check_collision(shape=cell_poly):
                        static_ids.append(static_idx)

                # find all dynamic obstacles in the current cell
                for dynamic_idx in self.dynamic_obstacles:
                    obstacle = self.dynamic_obstacles[dynamic_idx]
                    if obstacle.check_collision(shape=cell_poly):
                        dynamic_ids.append(dynamic_idx)

                static_arr[kx][ky] = static_ids
                dynamic_arr[kx][ky] = dynamic_ids

        return static_arr, dynamic_arr, spacing_x, spacing_y

    def plot(self, query_time: float = None, fig=None):
        """
        Plots the obstacles in the environment instance using matplotlib.

        Parameters:
        - query_time (float): The time at which the query is made (optional).
        """
        if fig is None:
            fig = plt.figure(figsize=(8, 8))

        # set the scenario boundaries as plotting boundaries
        plt.xlim(self.dim_x)
        plt.ylim(self.dim_y)

        # plot static obstacles independent of query time
        for obstacle_stat in self.static_obstacles.values():
            obstacle_stat.plot(fig=fig)

        # plot dynamic obstacles at the query time
        for obstacle_dyn in self.dynamic_obstacles.values():
            obstacle_dyn.plot(query_time=query_time, fig=fig)

    # TODO - add functions to compute the temporal availability of points and edges for PRM roadmap

    # def check_collision_static_pt(self, point: ShapelyPoint) -> bool:
    #     """
    #     Check if a given point is in collision with any static obstacle in the environment.

    #     Args:
    #         point (ShapelyPoint): The point to check for collision.

    #     Returns:
    #         bool: True if the point is in collision with any obstacle, False otherwise.
    #     """

    #     cell_x = math.floor((point.x - self.dim_x[0]) / sim.spacing_x)
    #     cell_y = math.floor((point.y - self.dim_y[0]) / sim.spacing_y)
    #     static_ids = self.static_idx[cell_x][cell_y]

    #     for key in static_ids:
    #         obstacle = self.static_obstacles[key]

    #         # if the point is in collision with any obstacle, return True
    #         if obstacle.check_collision(shape=point):
    #             return True

    #     # return False if no collision was found
    #     return False

    # def check_collision_dynamic_pt(
    #     self,
    #     point: ShapelyPoint,
    #     query_interval: Interval = None,
    #     query_time: float = None,
    # ) -> bool:
    #     """
    #     Check if a given point is in collision with any dynamic obstacle in the environment.

    #     Args:
    #         point (ShapelyPoint): The point to check for collision.
    #         query_interval (Interval, optional): The time interval for the collision query. Defaults to None.
    #         query_time (float, optional): The time for the collision query. Defaults to None.

    #     Returns:
    #         bool: True if the point is in collision with any dynamic obstacle, False otherwise.
    #     """
    #     cell_x = math.floor((point.x - self.dim_x[0]) / sim.spacing_x)
    #     cell_y = math.floor((point.y - self.dim_y[0]) / sim.spacing_y)
    #     dynamic_ids = self.dynamic_idx[cell_x][cell_y]

    #     for key in dynamic_ids:
    #         obstacle = self.dynamic_obstacles[key]

    #         # if the point is in collision with any obstacle, return True
    #         if obstacle.check_collision(
    #             shape=point, query_time=query_time, query_interval=query_interval
    #         ):
    #             return True

    #     # return False if no collision was found
    #     return False
