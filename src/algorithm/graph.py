import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import LineString as ShapelyLine

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
        vertices (dict): A dictionary that maps vertex indices to their corresponding coordinates.
        edges (dict): A dictionary that stores the edges between vertices.
        connections (dict): A dictionary that stores the connections between vertices for faster access.

    Methods:
        __init__: Initializes a Graph object.
        __sample_nodes: Generates random points as vertices in the graph.
        __connect_vertices: Connects vertices in the graph.
        plot: Plots the graph, including all vertices and edges.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        neighbour_distance: float = 10.0,
        max_connections: int = 10,
        env: EnvironmentInstance = None,
    ):
        """
        Initializes a Graph object.

        Args:
            num_samples (int): The number of samples to generate.
            neighbour_distance (float): The maximum distance between neighboring vertices.
            env (EnvironmentInstance): The environment instance to use for sampling and collision checking.
        """
        # save the environment instance
        self.env = env

        # save other parameters
        self.num_vertices = num_samples
        self.neighbour_distance = neighbour_distance
        self.max_connections = max_connections

        # sample random vertices
        self.__sample_nodes(num_samples)

        # connect vertices
        self.__connect_vertices()

    def __sample_nodes(self, num_samples: int):
        """
        Generates random points as vertices in the graph.

        Args:
            num_samples (int): The number of samples to generate.
            env (EnvironmentInstance): The environment instance to use for sampling and collision checking.
        """
        self.vertices = {}
        vertex_idx = 0

        while vertex_idx < num_samples:
            pt = self.env.sample_point()

            if not self.env.static_collision_free(pt):
                continue
            else:
                self.vertices[vertex_idx] = pt
                vertex_idx += 1

    def __connect_vertices(self):
        """
        Connects the vertices in the graph by creating edges between neighboring vertices.

        This method initializes the edges and connections dictionaries, and then iterates over each vertex in the graph.
        For each vertex, it finds all neighboring vertices within the specified distance and creates an edge between them.
        The edge is checked for collisions with static obstacles, and if it is collision-free, it is added to the graph.
        The corresponding availability with respect to dynamic obstacles is checked and influnces the availability of the edge.

        Returns:
            None
        """

        # initialize edges and connections
        # format {"edge_id": EdgeWithTemporalAvailability}
        self.edges = {}
        # format {"node_id": [("neighbor_id", "edge_id"), ...]}
        self.connections = {key: [] for key in self.vertices}

        # initialize edge index
        edge_idx = 1

        print("Connecting vertices in the graph...")
        for key in tqdm(self.vertices):
            # get vertex
            vertex = self.vertices[key]

            # initialize neighbours list
            neighbours = []

            # if the maximum number of connections is reached, skip
            if len(self.connections[key]) >= self.max_connections:
                continue

            # find all neighbours within the specified distance
            for other_key in self.vertices:
                other_vertex = self.vertices[other_key]
                if vertex.distance(other_vertex) <= self.neighbour_distance:
                    neighbours.append(other_key)

            # connect all neighbours within the maximum connection distance
            for nkey in neighbours:
                # skip if the neighbour is the same as the current vertex
                if nkey == key:
                    continue

                # if either one of the vertices has reached the maximum number of connections, skip
                if (
                    len(self.connections[key]) >= self.max_connections
                    or len(self.connections[nkey]) >= self.max_connections
                ):
                    continue

                # extract neighbour key and shapely coordinates
                nnode = self.vertices[nkey]

                # create edge and check it for collisions with static obstacles
                edge_candidate = ShapelyLine([(vertex.x, vertex.y), (nnode.x, nnode.y)])

                # if the edge is static collision free, add it to the graph
                ln_static_free, cells = self.env.static_collision_free_ln(
                    edge_candidate
                )

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
                        self.edges[edge_idx] = TimedEdge(
                            geometry=edge_candidate,
                            always_available=always_available,
                            availability=free_intervals,
                        )
                        self.connections[key].append((nkey, edge_idx))
                        self.connections[nkey].append((key, edge_idx))
                        edge_idx += 1

    # TODO - add methods to connect start and goal node to graph

    def plot(self, query_time: float = None, fig=None):
        """
        Plots the graph, including all vertices and edges.

        Parameters:
        - query_time (float): The time at which the query is made (optional).
        - fig (matplotlib.pyplot.figure): The figure to plot the obstacles on (optional).
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

    # TODO - add functions to save and load from file
