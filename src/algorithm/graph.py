import cv2
import datetime
import os
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import LineString as ShapelyLine, Point as ShapelyPoint
from typing import Tuple, List
import numpy as np
from scipy.stats import qmc

from src.envs.environment_instance import EnvironmentInstance
from src.algorithm.timed_edge import TimedEdge


class Graph:
    """
    Represents a graph used in the PRM algorithm.

    Args:
        num_samples (int): The number of samples to generate as vertices in the graph.
        neighbour_distance (float): The maximum distance between a vertex and its neighbor.
        env (EnvironmentInstance): An instance of the environment in which the graph is constructed.

    Attributes:
        env (EnvironmentInstance): The environment instance in which the graph is constructed.
        vertices (dict): A dictionary that maps vertex indices to their corresponding coordinates.
        edges (dict): A dictionary that stores the edges between vertices.
        connections (dict): A dictionary that stores the connections between vertices for faster access.
        num_vertices (int): The number of vertices in the graph.
        neighbour_distance (float): The maximum distance between a vertex and its neighbor.
        max_connections (int): The maximum number of connections for each vertex.
        start (int): The index of the start vertex.
        goal (int): The index of the goal vertex.

    Methods:
        __init__: Initializes a Graph object.
        __sample_nodes: Generates random points as vertices in the graph.
        connect_vertices: Connects vertices in the graph.
        __connect_neighbours: Connects the given vertex with its neighboring vertices within a specified distance.
        plot: Plots the graph, including all vertices and edges.
    """

    def __init__(
        self,
        env: EnvironmentInstance,
        num_samples: int = 1000,
        neighbour_distance: float = 10.0,
        max_connections: int = 10,
        seed: int = None,
        quiet: bool = False,
    ):
        """
        Initializes a Graph object.

        Args:
            env (EnvironmentInstance): The environment instance to use for sampling and collision checking.
            num_samples (int): The number of samples to generate.
            neighbour_distance (float): The maximum distance between neighboring vertices.
            max_connections (int): The maximum number of connections for each vertex.
            seed (int): The seed to use for random number generation.
            quiet (bool): If True, disables verbose print statements / progress bars.
        """
        # save the environment instance
        self.env = env

        # save other parameters
        self.num_vertices = num_samples
        self.neighbour_distance = neighbour_distance
        self.max_connections = max_connections

        # sample random vertices
        self.__sample_nodes(num_samples=num_samples, seed=seed)

        # connect vertices
        self.connect_vertices(quiet=quiet)

        # initialize empty start and goal vertex indices
        self.start = None
        self.goal = None

    def connect_start(self, coords: Tuple[float, float]):
        """
        Connects the start node to the graph.

        Args:
            coords (Tuple[float, float]): The coordinates of the start node.

        Raises:
            ValueError: If the start node is not collision-free or could not be connected to any other node.
        """
        # create shapely point
        start_pt = ShapelyPoint(coords[0], coords[1])

        # start and goal node cannot be the same
        if self.goal is not None and self.vertices[self.goal] == start_pt:
            raise ValueError("Start and goal node cannot be the same.")

        # check if start node is collision free
        if not self.env.static_collision_free(start_pt):
            raise ValueError("Start node is not collision free.")

        # extract index of start node, which will be inserted
        self.start = len(self.vertices)

        # add start node to vertices and create connect it to the graph
        self.vertices[self.start] = start_pt
        self.connections[self.start] = []
        success, _ = self.__connect_neighbours(
            self.start, next_edge_idx=len(self.edges), ignore_max_connections=True
        )

        # check if the start node was connected to any other node
        if not success or len(self.connections[self.start]) == 0:
            raise ValueError("Start node could not be connected to any other node.")

    def connect_goal(self, coords: ShapelyPoint, quiet: bool = False):
        """
        Connects the goal node to the graph.

        Args:
            coords (ShapelyPoint): The coordinates of the goal node.

        Raises:
            ValueError: If the goal node is not collision-free or could not be connected to any other node.
        """
        # create shapely point
        goal_pt = ShapelyPoint(coords[0], coords[1])

        # start and goal node cannot be the same
        if self.start is not None and self.vertices[self.start] == goal_pt:
            raise ValueError("Start and goal node cannot be the same.")

        # check if goal node is collision free
        if not self.env.static_collision_free(goal_pt):
            raise ValueError("Goal node is not collision free.")

        # extract index of goal node, which will be inserted
        self.goal = len(self.vertices)

        # add goal node to vertices and create connect it to the graph
        self.vertices[self.goal] = goal_pt
        self.connections[self.goal] = []
        success, _ = self.__connect_neighbours(
            self.goal, next_edge_idx=len(self.edges), ignore_max_connections=True
        )

        # check if the goal node was connected to any other node
        if not success or len(self.connections[self.goal]) == 0:
            raise ValueError("Goal node could not be connected to any other node.")

        # compute the heuristic cost-to-go values for all nodes
        if not quiet:
            print("Computing heuristic cost-to-go values...")

        for key in tqdm(self.vertices, disable=quiet):
            self.heuristic[key] = self.vertices[key].distance(self.vertices[self.goal])

    def __sample_nodes(self, num_samples: int, seed: int = None):
        """
        Generates random points as vertices in the graph.

        Args:
            num_samples (int): The number of samples to generate.
            env (EnvironmentInstance): The environment instance to use for sampling and collision checking.
        """
        # set up Halton sequence and draw samples
        lower_bounds = [self.env.dim_x[0], self.env.dim_y[0]]
        upper_bounds = [self.env.dim_x[1], self.env.dim_y[1]]
        sampler = qmc.Halton(d=2, scramble=False, seed=seed)

        self.vertices = {}
        vertex_idx = 0

        while vertex_idx < num_samples:
            # draw sample
            candidate = sampler.random(n=1)
            candidate = qmc.scale(candidate, lower_bounds, upper_bounds)

            pt = ShapelyPoint(candidate[0][0], candidate[0][1])

            if not self.env.static_collision_free(pt):
                continue
            else:
                self.vertices[vertex_idx] = pt
                vertex_idx += 1

    def connect_vertices(self, quiet: bool = False):
        """
        Connects the vertices in the graph by creating edges between neighboring vertices.

        This method initializes the edges and connections dictionaries, and then iterates over each vertex in the graph.
        For each vertex, it finds all neighboring vertices within the specified distance and creates an edge between them.
        The edge is checked for collisions with static obstacles, and if it is collision-free, it is added to the graph.
        The corresponding availability with respect to dynamic obstacles is checked and influnces the availability of the edge.

        Returns:
            None
        """

        # initialize edges, connections and heuristic dictionaries
        # format {"edge_id": EdgeWithTemporalAvailability}
        self.edges = {}
        # format {"node_id": [("neighbor_id", "edge_id"), ...]}
        self.connections = {key: [] for key in self.vertices}
        # format {"node_id": float}
        self.heuristic = {key: np.inf for key in self.vertices}

        # initialize edge index
        next_edge_idx = 0

        if not quiet:
            print("Connecting vertices in the graph...")

        for key in tqdm(self.vertices, disable=quiet):
            success, next_edge_idx = self.__connect_neighbours(
                key, next_edge_idx=next_edge_idx
            )

    def __connect_neighbours(
        self, vertex_idx: int, next_edge_idx: int, ignore_max_connections: bool = False
    ):
        """
        Connects the given vertex with its neighboring vertices within a specified distance.

        Args:
            vertex_idx (int): The index of the vertex to connect.

        Returns:
            bool: True if the vertex was successfully connected to at least one other vertex, False otherwise.
            int: The index of the next edge to be added to the graph.
        """

        # get vertex
        vertex = self.vertices[vertex_idx]

        # initialize neighbours list
        neighbours = []

        # if the maximum number of connections is reached, skip
        if (
            len(self.connections[vertex_idx]) >= self.max_connections
            and not ignore_max_connections
        ):
            return False, next_edge_idx

        # find all neighbours within the specified distance
        for other_key in self.vertices:
            other_vertex = self.vertices[other_key]
            distance_to_other = vertex.distance(other_vertex)

            if distance_to_other <= self.neighbour_distance:
                neighbours.append((other_key, distance_to_other))

        # sort neighbours by distance (ascending)
        neighbours = sorted(neighbours, key=lambda x: x[1])

        # track if node was successfully connected to any other node
        valid_connection = False

        # connect node to its closest neighbors
        for nkey, _ in neighbours:
            # skip if the neighbour is the same as the current vertex
            if nkey == vertex_idx:
                continue

            # skip if the neighbour is already connected to the current vertex
            if nkey in [x[0] for x in self.connections[vertex_idx]]:
                continue

            # if the current key has reached the maximum number of connections, skip
            if (
                len(self.connections[vertex_idx]) >= self.max_connections
                and not ignore_max_connections
            ):
                return True, next_edge_idx

            # if the neighbour node has reached the maximum number of connections, skip
            if (
                len(self.connections[nkey]) >= self.max_connections
                and not ignore_max_connections
            ):
                continue

            # extract neighbour key and shapely coordinates
            nnode = self.vertices[nkey]

            # create edge and check it for collisions with static obstacles
            edge_candidate = ShapelyLine([(vertex.x, vertex.y), (nnode.x, nnode.y)])

            # if the edge is static collision free, add it to the graph
            ln_static_free, cells = self.env.static_collision_free_ln(edge_candidate)

            if ln_static_free:
                (
                    always_available,
                    always_blocked,
                    free_intervals,
                ) = self.env.collision_free_intervals_ln(
                    line=edge_candidate, cells=cells
                )

                if always_blocked:
                    continue
                else:
                    # add edge to edges and update connections
                    self.edges[next_edge_idx] = TimedEdge(
                        geometry=edge_candidate,
                        always_available=always_available,
                        cost=edge_candidate.length,
                        availability=free_intervals,
                    )
                    self.connections[vertex_idx].append((nkey, next_edge_idx))
                    self.connections[nkey].append((vertex_idx, next_edge_idx))
                    valid_connection = True
                    next_edge_idx += 1

        return valid_connection, next_edge_idx

    def path_cost(self, sol_path: List[int] = None):
        """
        Calculates the cost of a given solution path.

        Args:
            sol_path (List[int]): The solution path represented as a list of node indices.

        Returns:
            float: The total cost of the solution path.
        """
        cost = 0
        for idx in range(len(sol_path) - 1):
            connections = self.connections[sol_path[idx]]
            for connection in connections:
                if connection[0] == sol_path[idx + 1]:
                    edge_idx = connection[1]
                    break

            cost += self.edges[edge_idx].cost

        return cost

    def plot(
        self,
        query_time: float = None,
        fig=None,
        sol_path: List[int] = None,
        quiet: bool = False,
    ):
        """
        Plots the graph, including all vertices and edges.

        Parameters:
        - query_time (float): The time at which the query is made (optional).
        - fig (matplotlib.pyplot.figure): The figure to plot the obstacles on (optional).
        - sol_path (List[int]): The solution path to plot (optional).
        """
        if fig is None:
            fig = plt.figure(figsize=(8, 8))

        # plot the environment instance
        self.env.plot(query_time=query_time, fig=fig)

        # plot vertices
        for vertex in self.vertices.values():
            plt.plot(vertex.x, vertex.y, color="blue", marker="o", markersize=1)

        # plot edges
        for line in self.edges.values():
            plt.plot(
                *line.geometry.xy,
                color="red",
                linewidth=0.5,
            )

        # plot solution path
        if sol_path is not None:
            if not quiet:
                cost = self.path_cost(sol_path=sol_path)
                print(f"Solution path cost: {cost:.2f}")

            for idx in range(len(sol_path) - 1):
                connections = self.connections[sol_path[idx]]
                for connection in connections:
                    if connection[0] == sol_path[idx + 1]:
                        edge_idx = connection[1]
                        break

                plt.plot(
                    *self.edges[edge_idx].geometry.xy,
                    color="green",
                    linewidth=3,
                )

        if self.start is not None:
            plt.plot(
                self.vertices[self.start].x,
                self.vertices[self.start].y,
                color="blue",
                marker="o",
                markersize=6,
            )

        if self.goal is not None:
            plt.plot(
                self.vertices[self.goal].x,
                self.vertices[self.goal].y,
                color="green",
                marker="o",
                markersize=6,
            )

    def simulate(
        self,
        start_time: float,
        sol_path: List[int],
        step: float = 0.25,
        fps: int = 10,
        filename: str = "simulation",
        plotting: bool = False,
        save_simulation: bool = False,
    ):
        """
        Simulates the movement along a given solution path in the graph.

        Parameters:
        - start_time (float): The starting time of the simulation.
        - sol_path (List[int]): The solution path represented as a list of vertex indices.
        - step (float, optional): The time step between each simulation iteration. Default is 0.25.
        - fps (int, optional): The frames per second for the simulation. Default is 10.
        - plotting (bool, optional): If True, the simulation will be shown. Default is False.
        - save_simulation (bool, optional): If True, the simulation will be saved as an mp4 file. Default is False.

        Returns:
        None
        """

        # 1) get the edge times to build a time-annotated path
        timed_path: List[Tuple(ShapelyPoint, float)] = [(sol_path[0], start_time)]

        for idx in range(len(sol_path) - 1):
            curr_vertex = sol_path[idx]
            curr_time = timed_path[-1][1]
            next_vertex = sol_path[idx + 1]
            connections = self.connections[curr_vertex]
            for connection in connections:
                if connection[0] == sol_path[idx + 1]:
                    edge_idx = connection[1]
                    break

            edge = self.edges[edge_idx]
            edge_time = edge.length
            timed_path.append((next_vertex, curr_time + edge_time))

        goal_time = timed_path[-1][1]

        # 2) simulate the path with dynamic querying
        fig = plt.figure(figsize=(8, 8))

        if save_simulation:
            # check if the folder "animations" exists in the current repository, otherwise create it
            if not os.path.exists("animations"):
                os.makedirs("animations")

            # create a temporary image directory
            if not os.path.exists("tmp_images"):
                os.makedirs("tmp_images")
            else:
                shutil.rmtree("tmp_images")
                os.makedirs("tmp_images")

            images = []

        print("Simulating solution path / creating simulation images...")
        for time in tqdm(np.arange(start_time, goal_time, step)):
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
            prev_coords = self.vertices[prev_vertex]
            next_coords = self.vertices[next_vertex]
            alpha = (time - prev_time) / (next_time - prev_time)
            curr_pos_x = prev_coords.x + alpha * (next_coords.x - prev_coords.x)
            curr_pos_y = prev_coords.y + alpha * (next_coords.y - prev_coords.y)

            # plot the environment with solution path at current simulation time
            plt.title(f"Simulation Time: {time}")
            self.plot(query_time=time, fig=fig, sol_path=sol_path, quiet=True)

            # plot the current position
            plt.plot(curr_pos_x, curr_pos_y, color="blue", marker="o", markersize=6)

            if plotting:
                plt.draw()
                plt.pause(0.001)
                plt.clf()

            if save_simulation:
                # save the current figure as an image
                image_path = f"tmp_images/image_{time}.png"
                plt.savefig(image_path, dpi=300)

                # save the image to the corresponding list
                images.append(image_path)

                # clear the figure
                plt.clf()

        if save_simulation:
            print("Creating simulation video...")
            current_time = datetime.datetime.now()
            mp4_path = (
                f"animations/{filename}_{current_time.strftime('%Y%m%d%H%M%S')}.avi"
            )
            frame = cv2.imread(images[0])
            height, width, layers = frame.shape

            video = cv2.VideoWriter(mp4_path, 0, fps, (width, height))

            for image in images:
                video.write(cv2.imread(image))

            cv2.destroyAllWindows()
            video.release()
            shutil.rmtree("tmp_images")

            # call ffmpeg from the command line to convert avi file to mp4 and compress size
            os.system(
                f"ffmpeg -i {mp4_path} -vcodec libx264 -crf 28 {mp4_path[:-4]}_compressed.mp4"
            )

            # delete avi file
            os.remove(mp4_path)
            print(f"Simulation saved as {mp4_path}")

    # TODO - add functions to save and load from file
