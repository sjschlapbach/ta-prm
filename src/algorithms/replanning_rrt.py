from typing import Tuple, List
from shapely.geometry import Point as ShapelyPoint

from src.envs.environment_instance import EnvironmentInstance
from src.algorithms.rrt import RRT


class ReplanningRRT:
    """Class representing the replanning framework for RRT / RRT*."""

    def __init__(
        self,
        env: EnvironmentInstance,
        seed: int = None,
    ):
        """
        Initialize the framework.

        Args:
            env (EnvironmentInstance): The environment instance.
            seed (int, optional): The random seed. Defaults to None.
        """
        self.env = env
        self.seed = seed

    def run(
        self,
        samples: int,
        stepsize: int,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        query_time: int,
        rewiring: bool,
        prev_path: list,
        dynamic_obstacles: bool,
        replannings: int = 0,
        return_on_replan_failure: bool = False,
        quiet: bool = False,
        in_recursion: bool = False,
        prev_paths: list = [],
    ):
        """
        Run the replanner with RRT / RRT* depending on the rewiring parameter.
        The function plans a path from start to goal using RRT / RRT*, considering all
        static and dynamic obstacles in the environment, which are visible at the time of
        the query. In case of a collision with a dynamic obstacle, the function triggers
        replanning from the last collision-free point with an updated query time.

        Args:
            samples (int): The number of samples to generate.
            stepsize (int): The step size for traversing the path.
            start (Tuple[int, int]): The starting point coordinates.
            goal (Tuple[int, int]): The goal point coordinates.
            query_time (int): The query time.
            rewiring (bool): Flag indicating whether to perform rewiring.
            prev_path (list): The previous path.
            dynamic_obstacles (bool): Flag indicating whether to consider dynamic obstacles.
            replannings (int, optional): The number of replannings. Defaults to 0.
            return_on_replan_failure (bool, optional): Flag indicating whether to return on replan failure. Defaults to False.
            quiet (bool, optional): Flag indicating whether to suppress output. Defaults to False.

        Returns:
            list: The final path.
            int: Total number of RRT planning runs.
        """
        # create tree - obstacles active at query_time will be considered as static obstacles
        if not quiet:
            print("Planning path...")

        rrt = RRT(
            start=start,
            goal=goal,
            env=self.env,
            num_samples=samples,
            query_time=query_time,
            seed=self.seed,
            rewiring=rewiring,
            consider_dynamic=dynamic_obstacles,
        )

        if not quiet:
            print("Found path with respect to all visible obstacles.")

        # compute solution path
        sol_path = rrt.rrt_find_path()

        # traverse path and check if recomputation is required along each edge with respect to the dynamic obstacles
        if not quiet:
            print("Validating path...")
        collision_free, save_idx, save_idx_time, last_save, last_time = (
            rrt.validate_path(
                path=sol_path, start_time=query_time, stepsize=stepsize, quiet=quiet
            )
        )

        if collision_free:
            if not quiet:
                print("Path is collision free.")
            final_path = prev_path + [
                rrt.tree[sol_path[idx]]["position"] for idx in range(1, len(sol_path))
            ]

            return final_path, replannings + 1, prev_paths

        else:
            if not quiet:
                print(
                    "Path is not collision free, with first collision at edge with starting point: ",
                    rrt.tree[sol_path[save_idx]]["position"],
                    "at time:",
                    save_idx_time,
                )

            # add all points up to the collision point to the final path (coordinates)
            new_path = prev_path + [
                rrt.tree[sol_path[i]]["position"] for i in range(1, save_idx + 1)
            ]

            if last_save is None:
                save_node = rrt.tree[sol_path[save_idx]]["position"]
                if (save_node.x, save_node.y) != start:
                    # if the last save node is not the start node, replan from the last save point
                    last_save = save_node
                    last_time = save_idx_time

                elif in_recursion:
                    # to skip issues where RRT fails due an obstacle popping up around a point
                    # currently considered to be save towards the goal
                    raise RuntimeError(
                        "Edge from new starting point is in collision on replanning."
                    )

                else:
                    raise RuntimeError(
                        "No collision-free point found on edge. Possibly, the step resolution is too large."
                    )

            if not quiet:
                print(
                    "Checked edge with collision, last save point: ",
                    last_save,
                    " at time: ",
                    last_time,
                    "--> triggering replanning...",
                )

            # add the collision point to the path
            new_path += [last_save]

            sol_nodes = [
                rrt.tree[sol_path[i]]["position"] for i in range(0, len(sol_path))
            ]
            prev_paths = prev_paths + [(sol_nodes, last_time)]

            # recompute path from last save point
            new_path, replannings, prev_paths = self.run(
                samples=samples,
                stepsize=stepsize,
                start=(last_save.x, last_save.y),
                goal=goal,
                query_time=last_time,
                rewiring=rewiring,
                prev_path=new_path,
                dynamic_obstacles=dynamic_obstacles,
                replannings=replannings,
                quiet=quiet,
                in_recursion=True,
                prev_paths=prev_paths,
            )

            return new_path, replannings + 1, prev_paths

    def simulate(
        self,
        start_time: float,
        sol_path: List[ShapelyPoint],
        stepsize: float = 1,
        waiting_time: float = 0.2,
    ):
        """
        Simulate the result of repeated planning.

        Args:
            start_time (float): The start time of the simulation.
            sol_path (List[ShapelyPoint]): The solution path.
            stepsize (float, optional): The step size for traversing the path. Defaults to 1.
            waiting_time (float, optional): The waiting time between steps. Defaults to 0.2.
        """
        # compute a time-annotated path
        timed_path: List[Tuple(ShapelyPoint, float)] = [(sol_path[0], start_time)]

        for idx in range(len(sol_path) - 1):
            curr_vertex = sol_path[idx]
            curr_time = timed_path[-1][1]
            next_vertex = sol_path[idx + 1]
            distance = curr_vertex.distance(next_vertex)

            timed_path.append((next_vertex, curr_time + distance))

        goal_time = timed_path[-1][1]

        # simulate the path in the environment
        self.env.simulate(
            start_time=start_time,
            goal_time=goal_time,
            sol_path=sol_path,
            timed_path=timed_path,
            stepsize=stepsize,
            waiting_time=waiting_time,
        )

    def get_path_cost(self, sol_path: List[ShapelyPoint]):
        """
        Compute the cost of the path.

        Args:
            sol_path (List[ShapelyPoint]): The solution path.

        Returns:
            float: The path cost.
        """
        cost = 0
        for i in range(len(sol_path) - 1):
            cost += sol_path[i].distance(sol_path[i + 1])

        return cost
