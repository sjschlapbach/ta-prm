from pandas import Interval
import numpy as np

from src.algorithms.rrt import RRT
from src.envs.environment import Environment
from src.envs.environment_instance import EnvironmentInstance


class RRTTest:
    def test_tree_creation(self):
        # time interval
        interval = Interval(interval_start, interval_end, closed="both")
        x_range = (0, 300)
        y_range = (0, 300)

        # create environment with random obstacles
        env = Environment()
        env.add_random_obstacles(
            num_points=100,
            num_lines=100,
            num_polygons=100,
            min_x=x_range[0],
            max_x=x_range[1],
            min_y=y_range[0],
            max_y=y_range[1],
            min_interval=interval.left,
            max_interval=interval.right,
            random_recurrence=True,
            seed=0,
        )

        # create environment instance
        env_inst = EnvironmentInstance(
            environment=env,
            query_interval=interval,
            scenario_range_x=x_range,
            scenario_range_y=y_range,
        )

        # create tree
        rrt = RRT(
            start=(2, 2),
            goal=(298, 298),
            env=env_inst,
            num_samples=300,
            seed=0,
        )

        assert len(rrt.tree.values()) == 300 + 2
        assert rrt.start == 0
        assert rrt.goal == 301
        assert rrt.tree[rrt.start]["position"].x == 2
        assert rrt.tree[rrt.start]["position"].y == 2
        assert rrt.tree[rrt.goal]["position"].x == 298
        assert rrt.tree[rrt.goal]["position"].y == 298
        assert rrt.tree[rrt.start]["parent"] is None
        assert rrt.tree[rrt.goal]["parent"] is not None
        assert len(rrt.tree[rrt.start]["children"]) > 0
        assert rrt.tree[rrt.goal]["children"] == []

        # if no error is thrown, a connection from start to goal should be available
        manual_path = [rrt.goal]
        current = rrt.goal
        while True:
            if current == rrt.start:
                break

            previous = rrt.tree[current]["parent"]
            assert previous is not None
            assert current in rrt.tree[previous]["children"]
            current = previous
            manual_path.append(current)

        sol_path = rrt.rrt_find_path()
        assert len(manual_path) > 1
        assert len(sol_path) > 1
        assert manual_path == sol_path
