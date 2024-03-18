import cv2
import datetime
import os
import json
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import LineString as ShapelyLine, Point as ShapelyPoint
from shapely import wkt
from typing import Tuple, List
import numpy as np
from scipy.stats import qmc

from src.envs.environment_instance import EnvironmentInstance
from src.algorithms.timed_edge import TimedEdge


class Graph:
    """
    Represents a graph used in the PRM algorithm.

    Args:
        num_samples (int): The number of samples to generate as vertices in the graph.
        env (EnvironmentInstance): An instance of the environment in which the graph is constructed.

    Attributes:
        env (EnvironmentInstance): The environment instance in which the graph is constructed.
        vertices (dict): A dictionary that maps vertex indices to their corresponding coordinates.
        edges (dict): A dictionary that stores the edges between vertices.
        connections (dict): A dictionary that stores the connections between vertices for faster access.
        num_vertices (int): The number of vertices in the graph.
        start (int): The index of the start vertex.
        goal (int): The index of the goal vertex.

    Methods:
        __init__: Initializes a Graph object.
        __connect_neighbours: Connects the given vertex with its neighboring vertices within a specified distance.
        plot: Plots the graph, including all vertices and edges.
        connect_start: Connects the start node to the graph.
        connect_goal: Connects the goal node to the graph.
        path_cost: Calculates the cost of a given solution path.
        simulate: Simulates the movement along a given solution path in the graph.
        save: Saves the graph to a file.
        load: Loads the graph from a file.
    """

    def __init__(
        self,
        env: EnvironmentInstance,
        num_samples: int = 1000,
        seed: int = None,
        quiet: bool = False,
        filename: str = None,
    ):
        """
        Initializes a Graph object.

        Args:
            env (EnvironmentInstance): The environment instance to use for sampling and collision checking.
            num_samples (int): The number of samples to generate.
            seed (int): The seed to use for random number generation.
            quiet (bool): If True, disables verbose print statements / progress bars.
        """
        # save the environment instance
        self.env = env

        if filename is not None:
            self.load(filename)
            return

        # save other parameters
        self.num_vertices = num_samples

        # if a seed is specified, set it
        if seed is not None:
            np.random.seed(seed)

        # initialize empty start and goal vertex indices
        self.start = None
        self.goal = None

        # initialize parameter required for sample-dependent connection distance
        d = 2
        obs_free_volume = env.get_static_obs_free_volume()
        unit_ball_volume = np.pi
        self.gammaPRM = (
            2
            * ((1 + 1 / d) ** (1 / d))
            * ((obs_free_volume / unit_ball_volume) ** (1 / d))
            + 1e-10
        )

        # sample random vertices and connect them to all neighbors in gammaPRM-dependent distance
        self.vertices = {}
        vertex_idx = 0

        # initialize edges, connections and heuristic dictionaries
        # format {"edge_id": TimedEdge}
        self.edges = {}
        # format {"node_id": [("neighbor_id", "edge_id"), ...]}, initialize empty
        self.connections = {}
        # format {"node_id": float}, initialize infinite
        self.heuristic = {}

        # initialize edge index
        next_edge_idx = 0

        if not quiet:
            print("Connecting vertices in the graph...")

        while vertex_idx < num_samples:
            # draw sample from random uniform distribution
            x_candidate = np.random.uniform(env.dim_x[0], env.dim_x[1])
            y_candidate = np.random.uniform(env.dim_y[0], env.dim_y[1])
            pt = ShapelyPoint(x_candidate, y_candidate)

            if not self.env.static_collision_free(pt):
                continue
            else:
                self.vertices[vertex_idx] = pt
                self.connections[vertex_idx] = []
                self.heuristic[vertex_idx] = np.inf

                # connect to all neighbors in vertex-set size-dependent distance
                n = vertex_idx + 1
                neighbor_distance = self.gammaPRM * (np.log(n) / n) ** (1 / 2)
                success, next_edge_idx = self.__connect_neighbours(
                    vertex_idx=vertex_idx,
                    neighbor_distance=neighbor_distance,
                    next_edge_idx=next_edge_idx,
                )

                if success or vertex_idx == 0:
                    vertex_idx += 1

    def connect_start(
        self, coords: Tuple[float, float], override_distance: float = None
    ):
        """
        Connects the start node to the graph.

        Args:
            coords (Tuple[float, float]): The coordinates of the start node.

        Raises:
            RuntimeError: If the start node is not collision-free or could not be connected to any other node.
        """
        # create shapely point
        start_pt = ShapelyPoint(coords[0], coords[1])

        # start and goal node cannot be the same
        if self.goal is not None and self.vertices[self.goal] == start_pt:
            raise RuntimeError("Start and goal node cannot be the same.")

        # check if start node is collision free
        if not self.env.static_collision_free(start_pt):
            raise RuntimeError("Start node is not collision free.")

        # extract index of start node, which will be inserted
        self.start = len(self.vertices)

        # add start node to vertices and create connect it to the graph
        self.vertices[self.start] = start_pt
        self.connections[self.start] = []
        n = self.start + 1
        neighbor_distance = self.gammaPRM * (np.log(n) / n) ** (1 / 2)
        success, _ = self.__connect_neighbours(
            vertex_idx=self.start,
            neighbor_distance=(
                neighbor_distance if override_distance is None else override_distance
            ),
            next_edge_idx=len(self.edges),
        )

        # check if the start node was connected to any other node
        if not success or len(self.connections[self.start]) == 0:
            raise RuntimeError("Start node could not be connected to any other node.")

    def connect_goal(
        self, coords: ShapelyPoint, quiet: bool = False, override_distance: float = None
    ):
        """
        Connects the goal node to the graph.

        Args:
            coords (ShapelyPoint): The coordinates of the goal node.
            quiet (bool): If True, disables verbose print statements / progress bars.

        Raises:
            RuntimeError: If the goal node is not collision-free or could not be connected to any other node.
        """
        # create shapely point
        goal_pt = ShapelyPoint(coords[0], coords[1])

        # start and goal node cannot be the same
        if self.start is not None and self.vertices[self.start] == goal_pt:
            raise RuntimeError("Start and goal node cannot be the same.")

        # check if goal node is collision free
        if not self.env.static_collision_free(goal_pt):
            raise RuntimeError("Goal node is not collision free.")

        # extract index of goal node, which will be inserted
        self.goal = len(self.vertices)

        # add goal node to vertices and create connect it to the graph
        self.vertices[self.goal] = goal_pt
        self.connections[self.goal] = []
        n = self.goal + 1
        neighbor_distance = self.gammaPRM * (np.log(n) / n) ** (1 / 2)
        success, _ = self.__connect_neighbours(
            vertex_idx=self.goal,
            neighbor_distance=(
                neighbor_distance if override_distance is None else override_distance
            ),
            next_edge_idx=len(self.edges),
        )

        # check if the goal node was connected to any other node
        if not success or len(self.connections[self.goal]) == 0:
            raise RuntimeError("Goal node could not be connected to any other node.")

        # compute the heuristic cost-to-go values for all nodes
        if not quiet:
            print("Computing heuristic cost-to-go values...")

        for key in tqdm(self.vertices, disable=quiet):
            self.heuristic[key] = self.vertices[key].distance(self.vertices[self.goal])

    def __connect_neighbours(
        self, vertex_idx: int, neighbor_distance: float, next_edge_idx: int
    ):
        """
        Connects the given vertex with its neighboring vertices within a specified distance.

        Args:
            vertex_idx (int): The index of the vertex to connect.
            neighbor_distance (float): The maximum distance at which to connect neighboring vertices.
            next_edge_idx (int): The index of the next edge to be added to the graph.

        Returns:
            bool: True if the vertex was successfully connected to at least one other vertex, False otherwise.
            int: The index of the next edge to be added to the graph.
        """

        # get vertex
        vertex = self.vertices[vertex_idx]

        # initialize neighbours list
        neighbours = []

        # find all neighbours within the specified distance
        for other_key in self.vertices:
            other_vertex = self.vertices[other_key]
            distance_to_other = vertex.distance(other_vertex)

            if distance_to_other <= neighbor_distance:
                neighbours.append((other_key, distance_to_other))

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
        show_inactive: bool = False,
        quiet: bool = False,
    ):
        """
        Plots the graph, including all vertices and edges.

        Parameters:
            query_time (float): The time at which the query is made (optional).
            fig (matplotlib.pyplot.figure): The figure to plot the obstacles on (optional).
            sol_path (List[int]): The solution path to plot (optional).
            show_inactive (bool): If True, inactive obstacles will be shown (optional).
            quiet (bool): If True, no additional information will be printed (optional).
        """
        if fig is None:
            fig = plt.figure(figsize=(8, 8))

        # plot the environment instance
        self.env.plot(query_time=query_time, show_inactive=show_inactive, fig=fig)

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
        show_inactive: bool = False,
    ):
        """
        Simulates the movement along a given solution path in the graph.

        Parameters:
            start_time (float): The starting time of the simulation.
            sol_path (List[int]): The solution path represented as a list of vertex indices.
            step (float, optional): The time step between each simulation iteration. Default is 0.25.
            fps (int, optional): The frames per second for the simulation. Default is 10.
            filename (str, optional): The name of the simulation file. Default is "simulation".
            plotting (bool, optional): If True, the simulation will be shown. Default is False.
            save_simulation (bool, optional): If True, the simulation will be saved as an mp4 file. Default is False.
            show_inactive (bool, optional): If True, inactive obstacles will be shown. Default is False.

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
            plt.title(f"Simulation Time: {round(time, 2)}")
            self.plot(
                query_time=time,
                fig=fig,
                sol_path=sol_path,
                show_inactive=show_inactive,
                quiet=True,
            )

            # plot the current position
            plt.plot(curr_pos_x, curr_pos_y, color="blue", marker="o", markersize=6)

            if plotting:
                plt.draw()
                plt.pause(step)
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

    def save(self, filename: str):
        """
        Saves the graph to a file.

        Args:
            filename (str): The name of the file to save the graph to.
        """

        json_object = {
            "num_vertices": self.num_vertices,
            "start": self.start,
            "goal": self.goal,
            "vertices": {key: vertex.wkt for key, vertex in self.vertices.items()},
            "edges": {key: edge.export_to_json() for key, edge in self.edges.items()},
            "connections": self.connections,
            "heuristic": self.heuristic,
        }

        with open(filename, "w") as f:
            json.dump(json_object, f)

    def load(self, filename: str):
        """
        Loads the graph from a file.

        Args:
            filename (str): The name of the file to load the graph from.
        """

        with open(filename, "r") as f:
            json_object = json.load(f)

        self.num_vertices = json_object["num_vertices"]
        self.start = json_object["start"]
        self.goal = json_object["goal"]
        self.vertices = {
            int(key): wkt.loads(value) for key, value in json_object["vertices"].items()
        }
        self.edges = {
            int(key): TimedEdge(geometry=None, availability=[], json_obj=value)
            for key, value in json_object["edges"].items()
        }
        self.connections = {
            int(key): value for key, value in json_object["connections"].items()
        }
        self.heuristic = {
            int(key): value for key, value in json_object["heuristic"].items()
        }
