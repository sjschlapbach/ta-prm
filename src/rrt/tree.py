from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point as ShapelyPoint, LineString as ShapelyLine

from src.envs.environment_instance import EnvironmentInstance


class Tree:
    """
    Represents a tree structure used in the RRT algorithm.

    Parameters:
    - root (Tuple[float, float]): The coordinates of the root / start node.
    - goal (Tuple[float, float]): The coordinates of the goal node.
    - env (EnvironmentInstance): An instance of the environment.
    - num_samples (int): The number of samples to build the tree.
    - seed (int): The seed for the random number generator.
    - quiet (bool): Whether to suppress output messages.

    Attributes:
    - tree (dict): A dictionary representing the tree structure.
    - start (int): The index of the start node in the tree.
    - goal (int): The index of the goal node in the tree.

    Methods:
    - __init__: Initializes the Tree object.
    - __find_closest_neighbor: Finds the closest neighbor to a given candidate node.
    - __check_connection_collision_free: Checks if the connection between a neighbor and a candidate node is collision-free.
    - plot: Plots the tree structure.
    """

    def __init__(
        self,
        root: Tuple[float, float],
        goal: Tuple[float, float],
        env: EnvironmentInstance,
        num_samples: int = 100,
        seed: int = None,
        quiet: bool = False,
    ):
        # check for collision of root node
        if not env.static_collision_free(ShapelyPoint(root[0], root[1])):
            raise ValueError("Root node is in collision with static obstacles.")

        # initialize tree
        self.tree = {
            0: {
                "position": ShapelyPoint(root[0], root[1]),
                "parent": None,
                "children": [],
            }
        }
        self.start = 0
        self.env = env
        next_sample = 2

        # build the tree up to the required number of samples
        while next_sample <= num_samples + 1:
            x_candidate = np.random.uniform(env.dim_x[0], env.dim_x[1])
            y_candidate = np.random.uniform(env.dim_y[0], env.dim_y[1])
            candidate = ShapelyPoint(x_candidate, y_candidate)

            # find closest neighbour
            neighbor, distance = self.__find_closest_neighbor(candidate=candidate)

            # check if edge is (static) collision-free, and if so, add the new node to the tree
            # dynamic obstacles are not considered during building phase of RRT graph
            if self.__check_connection_collision_free(
                neighbor=neighbor, candidate=candidate, distance=distance
            ):
                self.tree[next_sample] = {
                    "position": candidate,
                    "parent": neighbor,
                    "children": [],
                }
                self.tree[neighbor]["children"].append(next_sample)
                next_sample += 1

        # connect goal to the tree as well
        self.goal = next_sample
        goal_node = ShapelyPoint(goal[0], goal[1])
        neighbor, distance = self.__find_closest_neighbor(candidate=goal_node)

        if self.__check_connection_collision_free(
            neighbor=neighbor, candidate=goal_node, distance=distance
        ):
            self.tree[next_sample] = {
                "position": goal_node,
                "parent": neighbor,
                "children": [],
            }
            self.tree[neighbor]["children"].append(next_sample)
        else:
            raise ValueError(
                "Goal node is not reachable from the tree or not collision free."
            )

    def __find_closest_neighbor(self, candidate: ShapelyPoint) -> Tuple[int, float]:
        """
        Finds the closest neighbor to the given candidate point in the tree.

        Args:
            candidate (ShapelyPoint): The point for which to find the closest neighbor.

        Returns:
            Tuple[int, float]: The key of the closest neighbor node in the tree and the distance between the candidate point and the closest neighbor.
        """
        closest = None
        distance = np.inf

        for key, node in self.tree.items():
            dist = candidate.distance(node["position"])
            if dist < distance:
                closest = key
                distance = dist

        return closest, distance

    def __check_connection_collision_free(
        self, neighbor: int, candidate: ShapelyPoint, distance: float
    ):
        """
        Checks if the connection between the neighbor node and the candidate node is collision-free.

        Args:
            neighbor (int): Index of the neighbor node in the tree.
            candidate (ShapelyPoint): The candidate node to connect with the neighbor node.
            distance (float): The distance between the neighbor node and the candidate node.

        Returns:
            bool: True if the connection is collision-free, False otherwise.
        """
        neighbor_node = self.tree[neighbor]["position"]
        edge = ShapelyLine(
            [(neighbor_node.x, neighbor_node.y), (candidate.x, candidate.y)]
        )

        # check for static collisions
        collision_free, _ = self.env.static_collision_free_ln(edge)
        if not collision_free:
            return False
        else:
            return True

    def plot(
        self,
        query_time: float = None,
        fig=None,
        sol_path: List[int] = None,
        show_inactive: bool = False,
        quiet: bool = False,
    ):
        """
        Plot the RRT tree.

        Parameters:
        - query_time (float): The time at which the query is made (optional, only effects the environment instance plot).
        - fig: The figure object to plot on (optional).
        - sol_path (List[int]): The solution path to plot (optional).
        - show_inactive (bool): Whether to show inactive vertices (default: False).
        - quiet (bool): Whether to suppress output (default: False).
        """
        if fig is None:
            fig = plt.figure(figsize=(8, 8))

        # plot the environment instance
        self.env.plot(
            query_time=-1 if query_time is None else query_time,
            show_inactive=True,
            fig=fig,
        )

        # plot vertices
        for vertex in self.tree.values():
            plt.plot(
                vertex["position"].x,
                vertex["position"].y,
                color="blue",
                marker="o",
                markersize=1,
            )

        # plot edges
        for key, vertex in self.tree.items():
            if vertex["parent"] is not None:
                parent = self.tree[vertex["parent"]]["position"]
                plt.plot(
                    [parent.x, vertex["position"].x],
                    [parent.y, vertex["position"].y],
                    color="red",
                    linewidth=0.5,
                )

        # TODO: plot solution path if provided

        # plot start node
        if self.start is not None:
            plt.plot(
                self.tree[self.start]["position"].x,
                self.tree[self.start]["position"].y,
                color="blue",
                marker="o",
                markersize=6,
            )

        # plot goal node
        if self.goal is not None:
            plt.plot(
                self.tree[self.goal]["position"].x,
                self.tree[self.goal]["position"].y,
                color="green",
                marker="o",
                markersize=6,
            )
