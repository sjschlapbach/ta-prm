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
    replanning_rrt(
        samples=1000,
        env_inst=env_instance,
        start=start_coords,
        goal=goal_coords,
        query_time=0,
        seed=seed,
        rewiring=rrt_star,
        prev_path=[start_coords],
    )


def replanning_rrt(
    samples: int,
    env_inst: EnvironmentInstance,
    start: Tuple[float, float],
    goal: Tuple[float, float],
    query_time: float,
    seed: int = None,
    rewiring: bool = False,
    prev_path: List[tuple] = [],
):
    # create tree - obstacles active at query_time will be considered as static obstacles
    rrt = RRT(
        start=start,
        goal=goal,
        env=env_inst,
        num_samples=samples,
        query_time=query_time,
        seed=seed,
        rewiring=rewiring,
    )

    # compute solution path
    sol_path = rrt.rrt_find_path()

    # traverse path and check if recomputation is required along each edge with respect to the dynamic obstacles
    print("Validating path...")
    # TODO: evaluate alternative where path is validated stepwise with provided resolution (min obstacle width) and then returns the last save point as well
    collision_free, save_idx = rrt.validate_path(path=sol_path, start_time=query_time)

    if collision_free:
        print("Path is collision free.")
        final_path = prev_path + [
            rrt.tree[sol_path[idx]["position"]] for idx in range(1, len(sol_path))
        ]

    else:
        print(
            "Path is not collision free, with first collision at edge with starting point: ",
            sol_path[save_idx],
        )

        # TODO: compute collision point
        last_save_point = (0, 0)
        last_save_time = 0

        # add all points up to the collision point to the final path (coordinates)
        final_path = prev_path + [
            rrt.tree[sol_path[i]["position"]] for i in range(1, save_idx + 1)
        ]

        # add the collision point to the path
        final_path.append(last_save_point)

        # recompute path from last save point
        new_path = replanning_rrt(
            samples=samples,
            env_inst=env_inst,
            start=last_save_point,
            goal=goal,
            query_time=last_save_time,
            seed=seed,
            rewiring=rewiring,
            prev_path=final_path,
        )


if __name__ == "__main__":
    # run the RRT re-planning example (= trigger replanning in case of collision with dynamic obstacle)
    dynamic_rrt(rrt_star=False, seed=0)

    # run the RRT* re-planning example (= trigger replanning in case of collision with dynamic obstacle)
    dynamic_rrt(rrt_star=True, seed=0)
