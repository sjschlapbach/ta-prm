import matplotlib.pyplot as plt
from pandas import Interval
from typing import List, Tuple
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint

from src.envs.environment import Environment
from src.envs.environment_instance import EnvironmentInstance
from src.obstacles.polygon import Polygon
from src.algorithms.rrt import RRT


def dynamic_rrt(rrt_star: bool = False, seed: int = None):
    # create an environment with all types of obstacles (static and continuous)
    env = Environment()

    # add random obstacles to environment
    x_range = (0, 100)
    y_range = (0, 100)

    # add a large obstacle to the environment
    shapely_polygon = ShapelyPolygon([(30, 30), (70, 30), (70, 70), (30, 70)])
    polygon_obs = Polygon(
        geometry=shapely_polygon,
        time_interval=Interval(5, 100, closed="both"),
    )
    env.add_obstacles([polygon_obs])

    # create an environment from it
    env_instance = EnvironmentInstance(
        environment=env,
        query_interval=Interval(0, 200, closed="both"),
        scenario_range_x=x_range,
        scenario_range_y=y_range,
    )

    # plan path using RRT / RRT* depending on parameter, which can be set
    # create tree - obstacles active at query_time will be considered as static obstacles
    start_coords = (2, 2)
    goal_coords = (98, 98)

    # iterative planning with replanning triggered on collision
    path = replanning_rrt(
        samples=1000,
        env_inst=env_instance,
        stepsize=1,
        start=start_coords,
        goal=goal_coords,
        query_time=0,
        seed=seed,
        rewiring=rrt_star,
        prev_path=[ShapelyPoint(start_coords)],
        dynamic_obstacles=True,
    )

    # simulate the resulting path
    env_instance.simulate(start_time=0, sol_path=path, stepsize=1, waiting_time=0.1)

    return path


def replanning_rrt(
    samples: int,
    env_inst: EnvironmentInstance,
    stepsize: float,
    start: Tuple[float, float],
    goal: Tuple[float, float],
    query_time: float,
    seed: int = None,
    rewiring: bool = False,
    prev_path: List[tuple] = [],
    dynamic_obstacles: bool = True,
):
    # create tree - obstacles active at query_time will be considered as static obstacles
    print("Planning path...")
    rrt = RRT(
        start=start,
        goal=goal,
        env=env_inst,
        num_samples=samples,
        query_time=query_time,
        seed=seed,
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
        x_step = (next_node.x - save_node.x) / delta_distance
        y_step = (next_node.y - save_node.y) / delta_distance

        # track the position and time of the last node, which is not in collision
        last_save = None
        last_time = save_time

        # iterate over the edge and check
        for i in range(1, int(delta_distance / stepsize)):
            sample = ShapelyPoint(save_node.x + i * x_step, save_node.y + i * y_step)

            # check if the sample is in collision
            last_time += stepsize
            collision_free = env_inst.static_collision_free(
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
        new_path = replanning_rrt(
            samples=samples,
            env_inst=env_inst,
            stepsize=stepsize,
            start=(last_save.x, last_save.y),
            goal=goal,
            query_time=last_time,
            seed=seed,
            rewiring=rewiring,
            prev_path=new_path,
            dynamic_obstacles=dynamic_obstacles,
        )

        return new_path


if __name__ == "__main__":
    # run the RRT re-planning example (= trigger replanning in case of collision with dynamic obstacle)
    rrt_path = dynamic_rrt(rrt_star=False, seed=0)

    # run the RRT* re-planning example (= trigger replanning in case of collision with dynamic obstacle)
    rrt_star_path = dynamic_rrt(rrt_star=True, seed=0)
