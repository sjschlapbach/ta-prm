from typing import List, Union
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

        # TODO - test cases:
        # - no recurrence, before, during, after, covering, partial, intersecting beginning and end
        # - recurrence starting after, starting before and intersecting with first, covering with first
        #   ... covering with later one, intersecting with later one, intersecting with multiple, etc.abs

        self.static_obstacles: List[Union[Point, Line, Polygon]] = []
        self.dynamic_obstacles: List[
            Tuple[Recurrence, Union[Point, Line, Polygon]]
        ] = []

        print("Loading obstacles into environment instance...")
        for obstacle in tqdm(environment.obstacles):
            # if the obstacle is static, add it to the static obstacles
            if obstacle[1].time_interval is None:
                self.static_obstacles.append(obstacle[1])

            # save the query interval start and end
            query_start = query_interval.left
            query_end = query_interval.right

            # if the obstacle has no recurrence value
            if obstacle[0] == Recurrence.NONE:
                # if the obstacle is not active during the query interval, skip it
                if not obstacle[1].time_interval.overlaps(query_interval):
                    continue

                # check if the obstacle instance overlaps with the entire query interval
                max_start = max(obstacle[1].time_interval.left, query_start)
                min_end = min(obstacle[1].time_interval.right, query_end)

                # if the intersecting interval length is 0, skip the obstacle
                if max_start == min_end:
                    continue

                # create a copy of the obstacle
                obs_copy = obstacle[1].copy()

                # if the obstacle is active during the whole query interval, add it to the static obstacles
                if max_start == query_start and min_end == query_end:
                    self.static_obstacles.append(obs_copy)
                # otherwise, add it to the dynamic obstacles
                else:
                    intersection = Interval(max_start, min_end, closed="both")
                    self.dynamic_obstacles.append((intersection, obs_copy))

            # handle the cases where the obstacle has a recurrence value
            else:
                # skip any obstalces, which start after the query interval
                if obstacle[1].time_interval.left > query_end:
                    continue

                start_k = 0
                delta = query_start - obstacle[1].time_interval.left
                rec_length = obstacle[0].get_seconds()

                # if the obstacle starts after the query interval start, directly use the first occurence
                # otherwise, select the first recurring instance of the obstacle, which starts after the interval start
                if delta >= 0:
                    start_k = delta // rec_length

                # loop through the obstacles until the new interval does not overlap anymore
                while True:
                    # compute the start of the obstacle
                    obstacle_start = (
                        obstacle[1].time_interval.left + start_k * rec_length
                    )

                    # if the obstacle starts after the query interval end, stop the loop
                    if obstacle_start > query_end:
                        break

                    # compute the end of the obstacle
                    obstacle_end = obstacle_start + obstacle[1].time_interval.length

                    # if the obstacle ends before the query interval start, skip it
                    if obstacle_end < query_start:
                        start_k += 1
                        continue

                    # check if the obstacle instance overlaps with the query interval
                    max_start = max(obstacle_start, query_start)
                    min_end = min(obstacle_end, query_end)
                    if max_start < min_end:
                        # create a copy of the obstacle
                        obs_copy = obstacle[1].copy()

                        # if the intersecting interval is the same as the query interval, add it to the static obstacles
                        if max_start == query_start and min_end == query_end:
                            self.static_obstacles.append(obs_copy)
                        # otherwise, add it to the dynamic obstacles
                        else:
                            intersection = Interval(max_start, min_end, closed="both")
                            self.dynamic_obstacles.append((intersection, obs_copy))

                    start_k += 1

        print("Creating spatial index for environment instance...")
        # TODO - creating a spatial index (which also works for lines)
        print("TO BE IMPLEMENTED...")

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

    def __compute_intersection_interval(first: Interval, second: Interval):
        """
        Helper function to compute the intersection interval between two intervals.

        Parameters
        ----------
        first : Interval
            the first interval
        second : Interval
            the second interval

        Returns
        -------
        Interval
            the intersection interval between the two intervals
        """
        intersection_start = max(first.left, second.left)
        intersection_end = min(first.right, second.right)
        interval = Interval(intersection_start, intersection_end, closed="both")
        return interval
