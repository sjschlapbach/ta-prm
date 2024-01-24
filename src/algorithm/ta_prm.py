from typing import Tuple
from heapq import heappush, heappop, heapify
from pandas import Interval
import numpy as np

from src.algorithm.graph import Graph


class TAPRM:
    """
    The TA-PRM (Time-Aware Probabilistic Roadmap) algorithm class.

    This class represents an instance of the TA-PRM algorithm, which is used to plan a path from a start node to a goal node
    in a graph-based environment.

    Attributes:
        graph (Graph): The graph representing the environment.

    Methods:
        __init__(self, graph: Graph, start: Tuple[float, float] = None, goal: Tuple[float, float] = None):
            Initializes a TA-PRM algorithm instance.

        plan(self, start_time: float, logging: bool = False):
            Plans a path from the start node to the goal node using the TA-PRM algorithm.
    """

    def __init__(
        self,
        graph: Graph,
        start: Tuple[float, float] = None,
        goal: Tuple[float, float] = None,
    ):
        """
        Initializes a TA-PRM algorithm instance.

        Args:
            graph (Graph): The graph representing the environment.
            start (Tuple[float, float], optional): The start position. Defaults to None.
            goal (Tuple[float, float], optional): The goal position. Defaults to None.
        """
        self.graph = graph

        if start is not None:
            if self.graph.start is not None:
                raise ValueError("Start node already specified.")

            self.graph.connect_start(start)

        if goal is not None:
            if self.graph.goal is not None:
                raise ValueError("Goal node already specified.")

            self.graph.connect_goal(goal)

    def plan(self, start_time: float, logging: bool = False, quiet: bool = False):
        """
        Plans a path from the start node to the goal node using the TA-PRM algorithm.

        Args:
            start_time (float): The start time for the planning process.
            logging (bool, optional): Flag indicating whether to enable logging. Defaults to False.

        Returns:
            tuple: A tuple containing a boolean indicating whether a path was found and the path (vertex ids) itself.
        """

        # raise an error if start or goal node are not specified
        if self.graph.start is None or self.graph.goal is None:
            raise ValueError("Start or goal node not specified.")

        # initialize open list with start node - heapq sorts according to first element of tuple
        # tuples have the form (cost_to_come + heuristic, cost_to_come, node_idx, time, path)
        distance_start_goal = self.graph.vertices[self.graph.start].distance(
            self.graph.vertices[self.graph.goal]
        )
        open_list = [
            (
                0 + distance_start_goal,
                0,
                self.graph.start,
                start_time,
                [self.graph.start],
            )
        ]
        heapify(open_list)

        # track the maximum length of the open list over time
        max_open_list = 1

        while open_list:
            # track the maximum length of the open list over time
            max_open_list = max(max_open_list, len(open_list))

            node = heappop(open_list)

            if logging:
                print("Expanding node: ", node[2], " at time: ", node[3])

            # if the goal node is reached, return the path
            if node[2] == self.graph.goal:
                if not quiet:
                    print("Successfully found a path from start to goal.")
                return True, node[4], max_open_list

            # get all neighbours ids
            neighbours = self.graph.connections[node[2]]

            if logging:
                print("Found", len(neighbours), "neighbours.")

            # iterate over all neighbours
            for neighbour_id, edge_id in neighbours:
                # if a loop is detected, skip the neighbour
                if neighbour_id in node[4]:
                    continue

                # check if edge is available and skip it if not
                edge = self.graph.edges[edge_id]
                start_time = node[3]
                end_time = start_time + edge.length

                # TODO - think about pruning in cost dimension on top of this / or instead of this
                # if end point of the edge would only be reached after the scenario interval, skip it
                if end_time > self.graph.env.query_interval.right:
                    if logging:
                        print(
                            "Edge",
                            edge_id,
                            "would end after the scenario interval.",
                        )

                    continue

                edge_cost = edge.get_cost(Interval(start_time, end_time, closed="both"))

                # if edge is not available, skip it
                if np.isinf(edge_cost):
                    if logging:
                        print(
                            "Cost between nodes",
                            node[2],
                            "and",
                            neighbour_id,
                            "is infinite.",
                        )

                    continue
                else:
                    # add the new node to the open list
                    heuristic = self.graph.heuristic[neighbour_id]
                    cost_to_come = node[1] + edge.cost
                    cost = cost_to_come + heuristic
                    path = node[4] + [neighbour_id]
                    heappush(
                        open_list, (cost, cost_to_come, neighbour_id, end_time, path)
                    )

                    if logging:
                        print(
                            "Cost between nodes",
                            node[2],
                            "and",
                            neighbour_id,
                            "is",
                            edge_cost,
                        )
                        print(
                            "Added node",
                            neighbour_id,
                            "at time",
                            end_time,
                            "to open list.",
                        )
                        print(
                            "Current open list is (format - cost + heuristic, cost-to-come, id, time, path): "
                        )
                        print(open_list)

        raise ValueError(
            "No valid path found from start to goal within the specified scenario horizon."
        )
