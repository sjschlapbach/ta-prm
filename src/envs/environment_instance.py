from typing import List, Union, Dict
from pandas import Interval
from tqdm import tqdm
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

# ! idea: this class stores similar information as before, but
# only an obstacle list (no recurrence) and selects the correct
# ones based on the query interval (all which interesect with the query interval).

# ! additionally a spatial index is created for the obstacles,
# ! so that the closest obstacle can be found efficiently


class EnvironmentInstance:
    # TODO: Docstring

    def __init__(self, environment: Environment, query_interval: Interval):
        # TODO - docstring

        self.query_interval = query_interval
        self.static_obstacles: Dict[int, Union[Point, Line, Polygon]] = {}
        self.dynamic_obstacles: Dict[int, Union[Point, Line, Polygon]] = {}
        counter = 1

        print("Loading obstacles into environment instance...")
        for obstacle in tqdm(environment.obstacles):
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
                # print("Obstacle is not active during query interval")
                # print(obstacle.export_to_json())
                continue

        # print("Creating spatial index for environment instance...")
        # TODO - creating a spatial index (which also works for lines)
        # print("TO BE IMPLEMENTED...")

    # TODO - update to new obstacles structure and new spatial index
    def closest_polygon_distance(self, point: ShapelyPoint) -> float:
        """
        Computes the distance between a shapely point object and the closest polygon in the environment.
        If there is a collision between the point and a polygon, the distance returned will be 0.0.

        Parameters
        ----------
        point : Point
            a shapely point object

        Returns
        -------
        float
            the distance between the point and the closest polygon in the environment, or 0.0 if there is a collision
        """
        return min([polygon.distance(point) for polygon in self.polygons])

    # TODO - update to new obstacles structure and new spatial index
    def closest_line_distance(self, line: ShapelyLine) -> float:
        """
        Computes the distance between a shapely line object and the closest polygon in the environment.

        Parameters
        ----------
        line : LineString
            a shapely line object

        Returns
        -------
        float
            the distance between the line and the closest polygon in the environment
        """
        return min([line.distance(polygon) for polygon in self.polygons])
