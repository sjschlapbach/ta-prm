from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point as ShapelyPoint, LineString as ShapelyLine

from src.envs.environment_instance import EnvironmentInstance


class RRT:
    """
    Represents an instance of the RRT (RRT* if rewiring is enabled) algorithm.

    Parameters:
    - start (Tuple[float, float]): The coordinates of the start node.
    - goal (Tuple[float, float]): The coordinates of the goal node.
    - env (EnvironmentInstance): An instance of the environment.
    - num_samples (int): The number of samples to build the tree.
    - seed (int): The seed for the random number generator.
    - rewiring (bool): Whether to use the RRT* algorithm (default: False).
    - quiet (bool): Whether to suppress output messages.

    Attributes:
    - tree (dict): A dictionary representing the tree structure.
    - start (int): The index of the start node in the tree.
    - goal (int): The index of the goal node in the tree.

    Methods:
    - __init__: Initializes the RRT object.
    - __find_closest_neighbor: Finds the closest neighbor to a given candidate node.
    - __check_connection_collision_free: Checks if the connection between a neighbor and a candidate node is collision-free.
    - plot: Plots the tree structure.
    """

    def __init__(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        env: EnvironmentInstance,
        num_samples: int = 100,
        seed: int = None,
        rewiring: bool = False,
        quiet: bool = False,
    ):
        # check for collision of start node
        if not env.static_collision_free(ShapelyPoint(start[0], start[1])):
            raise ValueError("start node is in collision with static obstacles.")

        # initialize tree
        if rewiring:
            self.tree = {
                0: {
                    "position": ShapelyPoint(start[0], start[1]),
                    "cost": 0,
                    "parent": None,
                    "children": [],
                }
            }
        else:
            self.tree = {
                0: {
                    "position": ShapelyPoint(start[0], start[1]),
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

            # TODO: track distance to xnear alongside the node indices
            # find closest neighbour
            xnearest, distance, xnear = self.__find_closest_neighbor(
                candidate=candidate, rewiring=rewiring
            )

            # check if edge is (static) collision-free, and if so, add the new node to the tree
            # dynamic obstacles are not considered during building phase of RRT graph
            # TODO: extract this to function as the entire block is repeated for goal
            if self.__check_connection_collision_free(
                neighbor=xnearest, candidate=candidate
            ):
                # ! use standard RRT algorithm
                if not rewiring:
                    self.tree[next_sample] = {
                        "position": candidate,
                        "parent": xnearest,
                        "children": [],
                    }
                    self.tree[xnearest]["children"].append(next_sample)

                # ! use RRT* algorithm
                else:
                    xmin = xnearest
                    cmin = self.tree[xnearest]["cost"] + distance

                    for x in xnear:
                        new_cost = self.tree[x]["cost"] + self.tree[x][
                            "position"
                        ].distance(candidate)

                        if (
                            self.__check_connection_collision_free(
                                neighbor=x, candidate=candidate
                            )
                            and new_cost < cmin
                        ):
                            xmin = x
                            cmin = new_cost

                    self.tree[next_sample] = {
                        "position": candidate,
                        "cost": cmin,
                        "parent": xmin,
                        "children": [],
                    }
                    self.tree[xmin]["children"].append(next_sample)

                    # rewire all nodes in the vicinity of the new node
                    for x in xnear:
                        new_cost = cmin + self.tree[xmin]["position"].distance(
                            self.tree[x]["position"]
                        )

                        # if cost is improved and the new edge is collision-free, rewire the tree
                        if (
                            self.__check_connection_collision_free(
                                neighbor=next_sample,
                                candidate=self.tree[x]["position"],
                            )
                            and new_cost < self.tree[x]["cost"]
                        ):
                            prev_parent = self.tree[x]["parent"]
                            self.tree[prev_parent]["children"].remove(x)
                            self.tree[x]["parent"] = next_sample
                            self.tree[x]["cost"] = new_cost
                            self.tree[next_sample]["children"].append(x)

                # increment after successful sample addition
                next_sample += 1

        # connect goal to the tree as well
        self.goal = next_sample
        goal_node = ShapelyPoint(goal[0], goal[1])
        xnearest, distance, xnear = self.__find_closest_neighbor(
            candidate=goal_node, rewiring=rewiring
        )

        if self.__check_connection_collision_free(
            neighbor=xnearest, candidate=goal_node
        ):
            self.tree[next_sample] = {
                "position": goal_node,
                "parent": xnearest,
                "children": [],
            }
            self.tree[xnearest]["children"].append(next_sample)
        else:
            raise ValueError(
                "Goal node is not reachable from the tree or not collision free."
            )

    def rrt_find_path(self):
        """
        Finds the path from the start to the goal node in the tree.

        Returns:
            List[int]: A list of indices representing the path from the start to the goal node.
        """
        path = [self.goal]
        current = self.goal
        while current != self.start:
            current = self.tree[current]["parent"]
            path.append(current)

        return path[::-1]

    def __find_closest_neighbor(
        self, candidate: ShapelyPoint, rewiring: bool
    ) -> Tuple[int, float, List[int]]:
        """
        Finds the closest neighbor to the given candidate point in the tree.

        Args:
            candidate (ShapelyPoint): The point for which to find the closest neighbor.
            rewiring (bool): Whether to use the RRT* algorithm and also return all the neighbours in a radius of log(n) / n around the new candidate

        Returns:
            Tuple[int, float]: The key of the closest neighbor node in the tree and the distance between the candidate point and the closest neighbor.
        """
        closest = None
        distance = np.inf
        xnear = []

        for key, node in self.tree.items():
            dist = candidate.distance(node["position"])
            if dist < distance:
                closest = key
                distance = dist

        if rewiring:
            n = len(self.tree)
            for key, node in self.tree.items():
                dist = candidate.distance(node["position"])
                if dist < np.log(n) / n:
                    xnear.append(key)

        return closest, distance, xnear

    def __check_connection_collision_free(self, neighbor: int, candidate: ShapelyPoint):
        """
        Checks if the connection between the neighbor node and the candidate node is collision-free.

        Args:
            neighbor (int): Index of the neighbor node in the tree.
            candidate (ShapelyPoint): The candidate node to connect with the neighbor node.

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

        # plot solution path
        if sol_path is not None:
            for i in range(len(sol_path) - 1):
                parent = self.tree[sol_path[i]]["position"]
                child = self.tree[sol_path[i + 1]]["position"]
                plt.plot(
                    [parent.x, child.x],
                    [parent.y, child.y],
                    color="green",
                    linewidth=3,
                )

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
