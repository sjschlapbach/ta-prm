from typing import Tuple
from heapq import heappush, heappop, heapify
from pandas import Interval
import numpy as np

from src.algorithm.graph import Graph


class TAPRM:
    # TODO - docstring

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

    def plan(self, start_time: float, logging: bool = False):
        # TODO - docstring

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

        while open_list:
            node = heappop(open_list)

            if logging:
                print("Expanding node: ", node[2], " at time: ", node[3])

            # if the goal node is reached, return the path
            if node[2] == self.graph.goal:
                print("Successfully found a path from start to goal.")
                return True, node[4]

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
                    # TODO - check if the node at the current time is already in open
                    # TODO - update the node in open if the cost is smaller than before
                    # How can this be done efficiently?

                    # TODO - once check in open list is implemented, move this to else statement
                    # if the node is not already in the open list, add it to the open list
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

        raise ValueError("No valid path found from start to goal.")
