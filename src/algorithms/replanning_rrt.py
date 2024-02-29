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

        Returns:
            list: The final path.
        """
        # create tree - obstacles active at query_time will be considered as static obstacles
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
        print("Found path with respect to all visible obstacles.")

        # compute solution path
        sol_path = rrt.rrt_find_path()

        # traverse path and check if recomputation is required along each edge with respect to the dynamic obstacles
        print("Validating path...")
        collision_free, save_idx, save_time = rrt.validate_path(
            path=sol_path, start_time=query_time
        )

        if collision_free:
            print("Path is collision free.")
            final_path = prev_path + [
                rrt.tree[sol_path[idx]]["position"] for idx in range(1, len(sol_path))
            ]

            return final_path

        else:
            print(
                "Path is not collision free, with first collision at edge with starting point: ",
                rrt.tree[save_idx]["position"],
            )

            # add all points up to the collision point to the final path (coordinates)
            new_path = prev_path + [
                rrt.tree[sol_path[i]]["position"] for i in range(1, save_idx + 1)
            ]

            # follow the next edge with fixed size steps and save the last collision-free point
            save_node = new_path[-1]
            next_node = rrt.tree[sol_path[save_idx + 1]]["position"]
            delta_distance = next_node.distance(save_node)
            num_steps = int(delta_distance / stepsize)
            x_step = abs(next_node.x - save_node.x) / num_steps
            y_step = abs(next_node.y - save_node.y) / num_steps

            # track the position and time of the last node, which is not in collision
            last_save = None
            last_time = save_time

            # iterate over the edge and check
            for i in range(1, num_steps):
                sample = ShapelyPoint(
                    save_node.x + i * x_step, save_node.y + i * y_step
                )

                # check if the sample is in collision
                last_time += stepsize
                collision_free = self.env.static_collision_free(
                    point=sample, query_time=last_time
                )

                if collision_free:
                    last_save = sample
                    last_time = last_time
                else:
                    break

            if last_save is None:
                raise ValueError(
                    "No collision-free point found on edge. Possibly, the step resolution is too large."
                )

            print(
                "Checked edge with collision, last save point: ",
                last_save,
                " at time: ",
                last_time,
                "--> triggering replanning...",
            )

            # add the collision point to the path
            new_path += [last_save]

            # recompute path from last save point
            new_path = self.run(
                samples=samples,
                stepsize=stepsize,
                start=(last_save.x, last_save.y),
                goal=goal,
                query_time=last_time,
                rewiring=rewiring,
                prev_path=new_path,
                dynamic_obstacles=dynamic_obstacles,
            )

            return new_path

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
