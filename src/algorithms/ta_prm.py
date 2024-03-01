import time
from typing import Tuple
from heapq import heappush, heappop, heapify, _siftdown
from pandas import Interval
import numpy as np

from src.algorithms.graph import Graph


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
                raise RuntimeError("Start node already specified.")

            self.graph.connect_start(start)

        if goal is not None:
            if self.graph.goal is not None:
                raise RuntimeError("Goal node already specified.")

            self.graph.connect_goal(goal)

    def plan(
        self,
        start_time: float,
        logging: bool = False,
        timeout: float = None,
        quiet: bool = False,
    ):
        """
        Plans a path from the start node to the goal node using the TA-PRM algorithm.

        Args:
            start_time (float): The start time for the planning process.
            logging (bool, optional): Flag indicating whether to enable logging. Defaults to False.
            timeout (float, optional): The maximum time allowed for the planning process. Defaults to None.
            quiet (bool, optional): Flag indicating whether to suppress output. Defaults to False.

        Returns:
            tuple: A tuple containing a boolean indicating whether a path was found and the path (vertex ids) itself.
        """

        # raise an error if start or goal node are not specified
        if self.graph.start is None or self.graph.goal is None:
            raise RuntimeError("Start or goal node not specified.")

        # initialize open list with start node - heapq sorts according to first element of tuple
        # tuples have the form (cost_to_come + heuristic, cost_to_come, node_idx, time, path)
        open_list = [
            (
                0 + self.graph.heuristic[self.graph.start],
                0,
                self.graph.start,
                start_time,
                [self.graph.start],
            )
        ]
        heapify(open_list)
        expansions = 0

        # track the maximum length of the open list over time
        max_open_list = 1

        # set start time for timeout of planning procedure
        timeout_start = time.time() if timeout is not None else None

        while open_list and (
            timeout is None or (time.time() - timeout_start) < timeout
        ):
            # track the maximum length of the open list over time
            max_open_list = max(max_open_list, len(open_list))

            node = heappop(open_list)
            expansions += 1

            if logging:
                print("Expanding node: ", node[2], " at time: ", node[3])

            # if the goal node is reached, return the path
            if node[2] == self.graph.goal:
                if not quiet:
                    print("Successfully found a path from start to goal.")
                return True, node[4], max_open_list, expansions

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

        if timeout is not None and (time.time() - timeout_start) >= timeout:
            raise TimeoutError("Planning process timed out.")

        raise RuntimeError(
            "No valid path found from start to goal within the specified scenario horizon."
        )

    def plan_temporal(
        self,
        start_time: float,
        temporal_precision: int,
        logging: bool = False,
        timeout: float = None,
        quiet: bool = False,
    ):
        """
        Plans a path from the start node to the goal node using the TA-PRM algorithm. This version is extended
        with a temporal pruning / limited temporal precision setup, such that nodes are only reconsidered, if
        there are no other instances with closeby time stamps in the open list. Otherwise, the instance with
        smaller cost to come will be used (assuming constant heuristic values)

        Args:
            start_time (float): The start time for the planning process.
            temporal_precision (int): The number of decimal digits considered in the temporal dimension.
            logging (bool, optional): Flag indicating whether to enable logging. Defaults to False.
            timeout (float, optional): The maximum time allowed for the planning process. Defaults to None.
            quiet (bool, optional): Flag indicating whether to suppress output. Defaults to False.

        Returns:
            tuple: A tuple containing a boolean indicating whether a path was found and the path (vertex ids) itself.
        """

        # raise an error if start or goal node are not specified
        if self.graph.start is None or self.graph.goal is None:
            raise RuntimeError("Start or goal node not specified.")

        # initialize open list with start node - heapq sorts according to first element of tuple
        # tuples have the form (cost_to_come + heuristic, cost_to_come, node_idx, time, rounded_time, path)
        distance_start_goal = self.graph.heuristic[self.graph.start]

        # open list stores the full tuple
        rounded_start_time = round(start_time, temporal_precision)
        ol_idx = 0
        open_list = {
            ol_idx: (
                0 + distance_start_goal,
                0,
                self.graph.start,
                start_time,
                rounded_start_time,
                [self.graph.start],
            )
        }

        # open list heap stores the cost + heuristic and the index in the actual open list
        open_list_heap = [(0 + distance_start_goal, ol_idx)]
        heapify(open_list_heap)
        expansions = 0

        # create a hash list to find the key in the open list associated to a node at a certain time
        start_hash = hash((self.graph.start, rounded_start_time))
        open_list_hash = {start_hash: ol_idx}

        # track the maximum length of the open list over time
        max_open_list = 1

        # update the index of the next element to be added to the open list
        ol_idx += 1

        # set start time for timeout of planning procedure
        timeout_start = time.time() if timeout is not None else None

        while open_list and (
            timeout is None or (time.time() - timeout_start) < timeout
        ):
            # track the maximum length of the open list over time
            max_open_list = max(max_open_list, len(open_list))

            # get the node with the smallest cost + heuristic from the heap
            heap_node = heappop(open_list_heap)
            node_idx = heap_node[1]
            node = open_list[node_idx]
            expansions += 1

            # remove the node from the actual open list and the hashed list
            node_hash = hash((node[2], node[4]))
            del open_list[node_idx]
            del open_list_hash[node_hash]

            if logging:
                print("Expanding node: ", node[2], " at time: ", node[3])

            # if the goal node is reached, return the path
            if node[2] == self.graph.goal:
                if not quiet:
                    print("Successfully found a path from start to goal.")
                return True, node[5], max_open_list, expansions

            # get all neighbours ids
            neighbours = self.graph.connections[node[2]]

            if logging:
                print("Found", len(neighbours), "neighbours.")

            # iterate over all neighbours
            for neighbour_id, edge_id in neighbours:
                # if a loop is detected, skip the neighbour
                if neighbour_id in node[5]:
                    continue

                # check if edge is available and skip it if not
                edge = self.graph.edges[edge_id]
                start_time = node[3]
                end_time = start_time + edge.length

                # only consider edges ending within the scenario horizon
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
                    # pre-compute quantities for new node to be added
                    cost_to_come = node[1] + edge.cost
                    heuristic = self.graph.heuristic[neighbour_id]
                    cost = cost_to_come + heuristic
                    path = node[5] + [neighbour_id]
                    rounded_end_time = round(end_time, temporal_precision)

                    if open_list:
                        # hash the neighbour id and its rounded end time to check for containment in the open list
                        neighbour_hash = hash((neighbour_id, rounded_end_time))

                        # check if the neighbour node at a similar time is already in the open list
                        if neighbour_hash in open_list_hash:
                            # get the index of the neighbour node in the open list
                            neighbour_idx = open_list_hash[neighbour_hash]

                            # if the cost_to_come is smaller than for the node in the open list, update it
                            if cost_to_come < open_list[neighbour_idx][1]:
                                if logging:
                                    print(
                                        "Node",
                                        neighbour_id,
                                        "around time",
                                        end_time,
                                        "already in open list with larger cost und will be updated.",
                                    )

                                # store the old cost + heuristic for the update of the heap
                                old_cost = open_list[neighbour_idx][0]

                                # update the existing node in the open list
                                open_list[neighbour_idx] = (
                                    cost,
                                    cost_to_come,
                                    neighbour_id,
                                    end_time,
                                    rounded_end_time,
                                    path,
                                )

                                # find the element in the heap and update it
                                heap_idx = open_list_heap.index(
                                    (old_cost, neighbour_idx)
                                )
                                open_list_heap[heap_idx] = (cost, neighbour_idx)

                                # as the cost + heuristic has decreased, update the heap structure
                                _siftdown(open_list_heap, 0, heap_idx)

                            else:
                                # if the new cost is higher, do nothing
                                if logging:
                                    print(
                                        "Node",
                                        neighbour_id,
                                        "around time",
                                        end_time,
                                        "(within precision) already in open list with lower cost und will be skipped.",
                                    )

                                continue

                        else:
                            # add the new node to the open list
                            open_list[ol_idx] = (
                                cost,
                                cost_to_come,
                                neighbour_id,
                                end_time,
                                rounded_end_time,
                                path,
                            )

                            # update the open list heap
                            heappush(open_list_heap, (cost, ol_idx))

                            # update the open list hash map
                            node_hash = hash((neighbour_id, rounded_end_time))
                            open_list_hash[node_hash] = ol_idx

                            # increment the index for the next element to be added
                            ol_idx += 1

                    else:
                        # add the new node to the open list
                        open_list[ol_idx] = (
                            cost,
                            cost_to_come,
                            neighbour_id,
                            end_time,
                            rounded_end_time,
                            path,
                        )

                        # update the open list heap
                        heappush(open_list_heap, (cost, ol_idx))

                        # update the open list hash map
                        node_hash = hash((neighbour_id, rounded_end_time))
                        open_list_hash[node_hash] = ol_idx

                        # increment the index for the next element to be added
                        ol_idx += 1

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

        if timeout is not None and (time.time() - timeout_start) >= timeout:
            raise TimeoutError("Planning process timed out.")

        raise RuntimeError(
            "No valid path found from start to goal within the specified scenario horizon."
        )
