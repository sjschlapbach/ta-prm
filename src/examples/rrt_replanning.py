from pandas import Interval
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint

from src.envs.environment import Environment
from src.envs.environment_instance import EnvironmentInstance
from src.obstacles.polygon import Polygon
from src.algorithms.replanning_rrt import ReplanningRRT


def prepare_environment(
    rrt_star: bool = False, seed: int = None
) -> EnvironmentInstance:
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

    return env_instance


if __name__ == "__main__":
    env = prepare_environment()

    # set the start and goal coordinates
    start_coords = (2, 2)
    goal_coords = (98, 98)

    # initialize replanning RRT framework
    replanner = ReplanningRRT(env=env, seed=0)

    # setup problem
    samples = 1000
    stepsize = 1
    query_time = 0
    waiting_time = 0.1

    # run the RRT re-planning example (= trigger replanning in case of collision with dynamic obstacle)
    rrt_path = replanner.run(
        samples=samples,
        stepsize=stepsize,
        start=start_coords,
        goal=goal_coords,
        query_time=query_time,
        rewiring=False,
        prev_path=[ShapelyPoint(*start_coords)],
        dynamic_obstacles=True,
    )
    replanner.simulate(
        start_time=0, sol_path=rrt_path, stepsize=stepsize, waiting_time=waiting_time
    )

    # run the RRT re-planning example (= trigger replanning in case of collision with dynamic obstacle)
    rrt_path = replanner.run(
        samples=samples,
        stepsize=stepsize,
        start=start_coords,
        goal=goal_coords,
        query_time=query_time,
        rewiring=True,
        prev_path=[ShapelyPoint(*start_coords)],
        dynamic_obstacles=True,
    )
    replanner.simulate(
        start_time=0, sol_path=rrt_path, stepsize=stepsize, waiting_time=waiting_time
    )
