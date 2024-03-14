from typing import List, Union, Dict, Tuple
from pandas import Interval
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import (
    Polygon as ShapelyPolygon,
    Point as ShapelyPoint,
    LineString as ShapelyLine,
)
from shapely import union_all

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
        plot(self, query_time: float = None, show_inactive: bool = False, fig=None)
        static_collision_free(self, point: ShapelyPoint, query_time: float = None) -> bool
        static_collision_free_ln(self, line: ShapelyLine, query_time: float = None) -> Tuple[bool, List[Tuple[int, int]]]
        collision_free_intervals_ln(self, line: ShapelyLine, cells: List[Tuple[int, int]]) -> Tuple[bool, List[Interval]]
    """

    def __init__(
        self,
        environment: Environment,
        query_interval: Interval,
        scenario_range_x: Tuple[int, int],
        scenario_range_y: Tuple[int, int],
        resolution: int = 20,
        quiet: bool = False,
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

        if not quiet:
            print("Loading obstacles into environment instance...")

        for obstacle in tqdm(environment.obstacles, disable=quiet):
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

                raise RuntimeError(
                    "Obstacle was not added to any obstacle list, despite being active"
                )

            else:
                continue

        if not quiet:
            print("Creating spatial index for environment instance...")

        self.resolution = resolution
        (
            self.static_idx,
            self.dynamic_idx,
            self.spacing_x,
            self.spacing_y,
        ) = self.compute_spatial_indices(resolution)

        if not quiet:
            print("Environment instance created successfully!")

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

    def plot(self, query_time: float = None, show_inactive: bool = False, fig=None):
        """
        Plots the obstacles in the environment instance using matplotlib.

        Parameters:
        - query_time (float): The time at which the query is made (optional).
        - show_inactive (bool): A boolean value indicating if inactive dynamic obstacles should be plotted (optional).
        - fig (matplotlib.pyplot.figure): The figure to plot the obstacles on (optional).
        """
        if fig is None:
            fig = plt.figure(figsize=(8, 8))

        # set the scenario boundaries as plotting boundaries
        plt.xlim(self.dim_x)
        plt.ylim(self.dim_y)

        # plot static obstacles independent of query time
        for obstacle_stat in self.static_obstacles.values():
            obstacle_stat.plot(fig=fig, color="black", fill_color="blue", opacity=0.2)

        # plot dynamic obstacles at the query time
        for obstacle_dyn in self.dynamic_obstacles.values():
            obstacle_dyn.plot(
                query_time=query_time,
                fig=fig,
                color="black",
                fill_color="green",
                opacity=0.2,
                show_inactive=True,
            )

    def simulate(
        self,
        start_time: float,
        goal_time: float,
        sol_path: List[ShapelyPoint],
        timed_path: List[Tuple[ShapelyPoint, float]],
        stepsize: float = 1,
        waiting_time: float = 0.2,
    ):
        """
        Simulates a solution path in the environment instance using matplotlib.

        Parameters:
        - start_time (float): The starting time of the simulation.
        - goal_time (float): The goal time of the simulation.
        - sol_path (List[ShapelyPoint]): The solution path to simulate.
        - timed_path (List[Tuple(ShapelyPoint, float)]): The time-annotated path to simulate.
        - stepsize (float, optional): The time step size for the simulation. Defaults to 1.
        - waiting_time (float, optional): The waiting time between each plot update. Defaults to 0.2.
        """

        # create a figure
        fig = plt.figure(figsize=(8, 8))

        # iterate over time
        for time in np.arange(start_time, goal_time + 5, stepsize):
            # find the index and time at previous and next vertex along path
            prev_vertex = None
            next_vertex = None
            prev_time = None
            next_time = None

            for idx, (vertex, vertex_time) in enumerate(timed_path):
                if time >= vertex_time:
                    prev_vertex = vertex
                    prev_time = vertex_time

                    if prev_vertex == sol_path[-1]:
                        break
                    else:
                        next_vertex = timed_path[idx + 1][0]
                        next_time = timed_path[idx + 1][1]
                else:
                    break

            if next_vertex is None:
                if curr_vertex == sol_path[-1]:
                    print("Goal node has been reached by simulation")
                    break
                else:
                    print("Simulation failed")
                    break

            # linearly interpolate between vertices to find current position
            alpha = (
                (time - prev_time) / (next_time - prev_time)
                if next_time != prev_time
                else 0
            )
            curr_pos_x = prev_vertex.x + alpha * (next_vertex.x - prev_vertex.x)
            curr_pos_y = prev_vertex.y + alpha * (next_vertex.y - prev_vertex.y)

            # plot the environment with solution path at current simulation time
            plt.title(f"Simulation Time: {round(time, 2)}")
            self.plot(
                fig=fig,
                query_time=time,
                show_inactive=True,
            )

            # plot the solution path
            plt.plot(
                [point.x for point in sol_path],
                [point.y for point in sol_path],
                color="green",
                linewidth=3,
            )

            # plot the current position
            plt.plot(curr_pos_x, curr_pos_y, color="blue", marker="o", markersize=6)

            # show the plot
            plt.draw()
            plt.pause(waiting_time)
            plt.clf()

    def static_collision_free(
        self,
        point: ShapelyPoint,
        query_time: float = None,
        check_all_dynamic: bool = False,
    ) -> bool:
        """
        Check if a given point is in collision with any static obstacle in the environment.
        If a query time is specified, the dynamic obstacles active at this time will be considered to be
        visible to the algorithm and treated as static obstacles.

        Args:
            shape (ShapelyPoint): The point to check for collision.
            query_time (float, optional): The time at which the query is made. Dynamic obstacles at this time are considered to be visible and treated as static obstacles. Defaults to None.
            check_all_dynamic (bool, optional): A boolean value indicating if all dynamic obstacles should be checked for collision independent of the specification of a query_time. Defaults to False.

        Returns:
            bool: False if the point is in collision with any obstacle, True otherwise.
        """
        cell_x = math.floor((point.x - self.dim_x[0]) / self.spacing_x)
        cell_y = math.floor((point.y - self.dim_y[0]) / self.spacing_y)
        static_ids = self.static_idx[cell_x][cell_y]

        for key in static_ids:
            obstacle = self.static_obstacles[key]

            # if the point is in collision with any obstacle, return False
            if obstacle.check_collision(shape=point):
                return False

        if query_time is not None or check_all_dynamic:
            dynamic_ids = self.dynamic_idx[cell_x][cell_y]

            for key in dynamic_ids:
                obstacle = self.dynamic_obstacles[key]

                if obstacle.is_active(query_time=query_time):
                    if obstacle.check_collision(shape=point):
                        return False

        # return True if no collision was found
        return True

    def static_collision_free_ln(
        self, line: ShapelyLine, query_time: float = None
    ) -> Tuple[bool, List[Tuple[int, int]]]:
        """
        Checks if a line is collision-free with respect to the static obstacles in the environment.
        If a query time is specified, the dynamic obstacles active at this time will be considered to be
        visible to the algorithm and treated as static obstacles.

        Args:
            line (ShapelyLine): The line to check for collision.
            query_time (float, optional): The time at which the query is made. Dynamic obstacles at this time are considered to be visible and treated as static obstacles. Defaults to None.

        Returns:
            tuple: A tuple containing a boolean value indicating if the line is collision-free,
                   and a list of index cells, where potential obstacles should be considered.
        """

        # compute the cells, the line is in collision with
        collision_cells = self.__compute_collision_cells(line)

        # find all ids of static obstacles in the collision_cells
        static_ids = set()
        for cell in collision_cells:
            static_ids.update(self.static_idx[cell[0]][cell[1]])

        # if no obstacles were found in the considered cells, return True
        if len(static_ids) == 0 and query_time is None:
            return True, collision_cells

        # if static obstacles were found, check if the line is in collision with any of them
        for key in static_ids:
            obstacle = self.static_obstacles[key]

            # if the line is in collision with any obstacle, return False
            if obstacle.check_collision(shape=line):
                return False, []

        # if dynamic obstacles were found, check if they are active at query time and if so, if the line is in collision with any of them
        if query_time is not None:
            dynamic_ids = set()

            for cell in collision_cells:
                dynamic_ids.update(self.dynamic_idx[cell[0]][cell[1]])

            for key in dynamic_ids:
                obstacle = self.dynamic_obstacles[key]

                if obstacle.is_active(query_time=query_time):
                    if obstacle.check_collision(shape=line):
                        return False, []

        # return True if no collision was found
        return True, collision_cells

    def __compute_collision_cells(self, line: ShapelyLine):
        """
        Compute the cells, the line is geometrically in collision with.

        Args:
            line (ShapelyLine): The line to check for collision.

        Returns:
            List[Tuple[int, int]]: A list of index cells, where potential obstacles should be considered.
        """

        line_coords = list(line.coords)
        cell_x1 = math.floor((line_coords[0][0] - self.dim_x[0]) / self.spacing_x)
        cell_y1 = math.floor((line_coords[0][1] - self.dim_y[0]) / self.spacing_y)
        cell_x2 = math.floor((line_coords[1][0] - self.dim_x[0]) / self.spacing_x)
        cell_y2 = math.floor((line_coords[1][1] - self.dim_y[0]) / self.spacing_y)

        # find the cells, the line is actually in collision with
        collision_cells = []
        cells_minx = min(cell_x1, cell_x2)
        cells_maxx = max(cell_x1, cell_x2)
        cells_miny = min(cell_y1, cell_y2)
        cells_maxy = max(cell_y1, cell_y2)

        for kx in range(cells_minx, cells_maxx + 1):
            for ky in range(cells_miny, cells_maxy + 1):
                # create a polygon for the current cell
                x1 = self.dim_x[0] + kx * self.spacing_x
                x2 = self.dim_x[0] + (kx + 1) * self.spacing_x
                y1 = self.dim_y[0] + ky * self.spacing_y
                y2 = self.dim_y[0] + (ky + 1) * self.spacing_y
                cell_poly = ShapelyPolygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])

                # check if the line is in collision with the cell
                if cell_poly.intersects(line):
                    collision_cells.append((kx, ky))

        return collision_cells

    def collision_free_intervals_ln(
        self, line: ShapelyLine, cells: List[Tuple[int, int]]
    ) -> Tuple[bool, List[Interval]]:
        """
        Calculates the collision-free intervals along a given line segment within the specified cells.

        Args:
            line (ShapelyLine): The line segment to check for collisions.
            cells (List[Tuple[int, int]]): The cells to consider for dynamic obstacles.

        Returns:
            bool: A boolean value indicating if the line is collision-free.
            bool: A boolean value indicating if the line is always blocked
            List[Interval]: A list of collision-free intervals along the line segment.
        """

        query_start = self.query_interval.left
        query_end = self.query_interval.right

        # if the environment constains no dynamic obstacles or no cells were specified, return the entire query interval
        if len(self.dynamic_obstacles) == 0 or len(cells) == 0:
            return True, False, [Interval(query_start, query_end, closed="both")]

        # collect all dynamic obstacle ids in the selected cells
        dyn_ids = set()
        for cell in cells:
            dyn_ids.update(self.dynamic_idx[cell[0]][cell[1]])

        # if no dynamic obstacles were found, return the entire query interval
        if len(dyn_ids) == 0:
            return True, False, [Interval(query_start, query_end, closed="both")]

        # collect all start and end times of the dynamic obstacles
        start_times = []
        end_times = []

        # iterate over all active dynamic obstacles
        for dyn_id in dyn_ids:
            obstacle = self.dynamic_obstacles[dyn_id]

            # check if the obstacle is spatially in collision with the line
            if not obstacle.check_collision(shape=line):
                continue

            if obstacle.recurrence == Recurrence.NONE:
                start_times.append(obstacle.time_interval.left)
                end_times.append(obstacle.time_interval.right)
            else:
                # find the occurence covered by the query_interval
                delta_start = self.query_interval.left - obstacle.time_interval.left
                delta_end = self.query_interval.right - obstacle.time_interval.left
                recurrence_length = obstacle.recurrence.get_seconds()
                start_k = int(delta_start // recurrence_length)
                end_k = int(delta_end // recurrence_length)

                # if only one occurence intersects with the query interval, append its start and end times
                if end_k - start_k == 0:
                    start_times.append(
                        obstacle.time_interval.left + start_k * recurrence_length
                    )
                    end_times.append(
                        obstacle.time_interval.right + end_k * recurrence_length
                    )

                else:
                    # check if the first occurence intersects with the query interval
                    if self.query_interval.overlaps(
                        Interval(
                            obstacle.time_interval.left + start_k * recurrence_length,
                            obstacle.time_interval.right + start_k * recurrence_length,
                            closed="both",
                        )
                    ):
                        start_times.append(
                            obstacle.time_interval.left + start_k * recurrence_length
                        )
                        end_times.append(
                            obstacle.time_interval.right + start_k * recurrence_length
                        )

                    # add the remaining intersecting occurences to the start and end times
                    for k in range(start_k + 1, end_k + 1):
                        start_times.append(
                            obstacle.time_interval.left + k * recurrence_length
                        )
                        end_times.append(
                            obstacle.time_interval.right + k * recurrence_length
                        )

        # sort start and end times of intersecting obstacle occurences in ascending order
        start_times.sort()

        # counters to keep track of
        start_idx = 0
        end_idx = 0
        active_count = 0

        # count obstacles, which are active at the start of the query interval
        for time in start_times:
            if time <= query_start:
                active_count += 1
                start_idx += 1
            else:
                break

        # set the first free interval start to the query start if no obstacles are active
        interval_start = query_start if active_count == 0 else -1

        # combine start and end times into a single list and sort it
        start_times = [(time, "start") for time in start_times[start_idx:]]
        end_times = [(time, "end") for time in end_times]
        time_list = start_times + end_times
        time_list.sort(key=lambda tup: tup[0])

        # iterate over the remaining start and end times and find the collision free intervals
        intervals = []

        for time in time_list:
            # if the current time is after the query interval, stop the iteration
            if time[0] > query_end:
                break

            if time[1] == "start":
                # if no obstacle was active before, end the free interval
                if active_count == 0 and interval_start < time[0]:
                    intervals.append(Interval(interval_start, time[0], closed="both"))
                    interval_start = -1

                # increment the count of active obstacles
                active_count += 1

            else:
                # decrement the count of active obstacles
                active_count -= 1

                # if no obstacle is active after, start a new free interval
                if active_count == 0:
                    interval_start = time[0]

        # if no obstacle was active after the query interval, end the last free interval
        if active_count == 0 and interval_start != -1 and interval_start < query_end:
            intervals.append(Interval(interval_start, query_end, closed="both"))

        if len(intervals) == 0:
            return False, True, []
        if len(intervals) == 1 and intervals[0] == self.query_interval:
            return True, False, intervals
        else:
            return False, False, intervals

    def dynamic_collision_free_ln(
        self,
        line: ShapelyLine,
        query_interval: Interval,
        stepsize: float = 1,
        quiet: bool = False,
    ) -> bool:
        """
        Check if a given line is in collision with any dynamic obstacle in the environment.

        Args:
            line (ShapelyLine): The line to check for collision.
            query_interval (Interval): The time interval for which the obstacles are active.
            stepsize (float, optional): The step size for traversing the path. Defaults to 1.
            quiet (bool, optional): A boolean value indicating if the function should print progress information. Defaults to False.

        Returns:
            bool: False if the line is in collision with any obstacle, True otherwise.
        """
        # compute the cells, the line is in collision with
        collision_cells = self.__compute_collision_cells(line)

        # find all ids of dynamic obstacles in the collision_cells
        dynamic_ids = set()
        for cell in collision_cells:
            dynamic_ids.update(self.dynamic_idx[cell[0]][cell[1]])

        # if no dynamic obstacles were found in the considered cells, return True
        if len(dynamic_ids) == 0:
            return True, None, None

        # if dynamic obstacles were found, check if the line is in collision with any of them
        for key in dynamic_ids:
            obstacle = self.dynamic_obstacles[key]

            if obstacle.is_active(query_interval=query_interval):
                if obstacle.check_collision(shape=line):
                    # The line overlaps in time and space with obstacle -> POSSIBLE COLLISION
                    # --> verify collision with subsampling

                    # keep track of last save subsample and time
                    last_save = None
                    last_save_time = query_interval.left

                    # extract start and end points from line
                    start = ShapelyPoint(line.coords[0][0], line.coords[0][1])
                    end = ShapelyPoint(line.coords[1][0], line.coords[1][1])

                    # apply subsampling to check for collision
                    true_positive = False
                    delta_distance = end.distance(start)
                    num_steps = int(delta_distance / stepsize)
                    x_step = (end.x - start.x) / num_steps
                    y_step = (end.y - start.y) / num_steps

                    for i in range(1, num_steps):
                        sample = ShapelyPoint(
                            start.x + i * x_step, start.y + i * y_step
                        )
                        sample_time = last_save_time + stepsize

                        # check if the sample is in collision
                        collision_free = self.static_collision_free(
                            point=sample, query_time=sample_time
                        )

                        if collision_free:
                            last_save = sample
                            last_save_time = sample_time
                        else:
                            true_positive = True
                            break

                    # edge is in collision at some intermediary position
                    if true_positive:
                        return False, last_save, last_save_time

                    else:
                        # check if the end of the edge is in collision
                        collision_free = self.static_collision_free(
                            point=end, query_time=query_interval.right
                        )
                        if not collision_free:
                            return False, end, query_interval.right
                        elif not quiet:
                            print(
                                "Detected false positive in dynamic edge collision check with subsampling."
                            )

        return True, None, None

    def get_static_obs_free_volume(self):
        """
        Get the free-space volume with respect to the static obstacles in the environment.
        """

        # compute the union object of all static obstacles
        buffered_obs = [
            obstacle.geometry.buffer(obstacle.radius)
            for obstacle in self.static_obstacles.values()
        ]
        union = union_all(buffered_obs)

        # compute the free space of the entire environment instance
        env_area = (self.dim_x[1] - self.dim_x[0]) * (self.dim_y[1] - self.dim_y[0])
        free_space = env_area - union.area

        # return the free-space
        return free_space
