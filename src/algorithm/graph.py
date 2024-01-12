import matplotlib.pyplot as plt

from src.envs.environment_instance import EnvironmentInstance


class Graph:
    """
    Represents a graph used in the PRM algorithm.

    Args:
        num_samples (int): The number of samples to generate as vertices in the graph.
        max_neighbours (int): The maximum number of neighbors each vertex can have.
        neighbour_distance (float): The maximum distance between a vertex and its neighbor.
        env (EnvironmentInstance): An instance of the environment in which the graph is constructed.

    Attributes:
        vertices (dict): A dictionary that maps vertex indices to their corresponding coordinates.
        edges (dict): A dictionary that stores the edges between vertices.
        connections (dict): A dictionary that stores the connections between vertices for faster access.

    Methods:
        __init__: Initializes a Graph object.
        __sample_nodes: Generates random points as vertices in the graph.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        max_neighbours: int = 10,
        neighbour_distance: float = 10.0,
        env: EnvironmentInstance = None,
    ):
        """
        Initializes a Graph object.

        Args:
            num_samples (int): The number of samples to generate.
            max_neighbours (int): The maximum number of neighbors for each vertex.
            neighbour_distance (float): The maximum distance between neighboring vertices.
            env (EnvironmentInstance): The environment instance to use for sampling and collision checking.
        """
        # save the environment instance
        self.env = env

        # sample random vertices
        self.__sample_nodes(num_samples, env)

        # TODO - create edges between vertices and store them
        # TODO - additionally use list representation to have fast querying
        self.edges = {}
        self.connections = {}  # format {"node_id": [("neighbor_id", "edge_id"), ...]}

    def __sample_nodes(self, num_samples: int, env: EnvironmentInstance):
        """
        Generates random points as vertices in the graph.

        Args:
            num_samples (int): The number of samples to generate.
            env (EnvironmentInstance): The environment instance to use for sampling and collision checking.
        """
        self.vertices = {}
        vertex_idx = 0

        while vertex_idx < num_samples:
            pt = env.sample_point()

            if not env.static_collision_free(pt):
                continue
            else:
                self.vertices[vertex_idx] = pt
                vertex_idx += 1

    # TODO - add plotting functions and save and load from file

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

        # TODO - add plotting of the edges once implemented
