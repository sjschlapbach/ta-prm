import os
import cv2
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from pandas import Interval
from typing import List, Tuple
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint

from src.envs.environment import Environment
from src.envs.environment_instance import EnvironmentInstance
from src.obstacles.polygon import Polygon
from src.algorithms.graph import Graph
from src.algorithms.ta_prm import TAPRM
from src.algorithms.replanning_rrt import ReplanningRRT
from src.evaluation.scenario_illustration import (
    get_timed_path,
    get_timed_path_rrt,
    get_current_pos_timed_path,
    plot_taprm_path,
    plot_rrt_path,
)


def create_video(
    filename: str,
    tmin: float,
    tmax: float,
    step: float,
    fps: int,
    high_res: bool = False,
    highest_res: bool = False,
):
    print("Creating simulation video...")

    # fetch all required images for the generation of the video
    images = []
    for t in np.arange(tmin, tmax, step):
        t = round(t, 1)
        if high_res:
            image_path = f"animations/scenario_images/scenario_{t}_high_res.png"
        elif highest_res:
            image_path = f"animations/scenario_images/scenario_{t}_highest_res.png"
        else:
            image_path = f"animations/scenario_images/scenario_{t}.png"
        images.append(image_path)

    # concatenate images to video
    mp4_path = f"{filename}.avi"
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(mp4_path, 0, fps, (width, height))

    for image in images:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

    # call ffmpeg from the command line to convert avi file to mp4 and compress size
    os.system(
        f"ffmpeg -i {mp4_path} -vcodec libx264 -crf 28 {mp4_path[:-4]}_compressed.mp4"
    )

    # delete avi file
    os.remove(mp4_path)
    print(f"Simulation saved as {mp4_path}")


if __name__ == "__main__":
    print("Creating scenario video and images...")

    # ! Settings which steps to execute
    generate_setup_images = False
    generate_images = False
    generate_videos = False

    if not generate_setup_images and not generate_images and not generate_videos:
        print("No steps selected for execution. Exiting...")
        exit()

    # ! Basic seed for reproducibility
    seed = 34875

    # ! Plotting settings
    plotting_start = 0
    plotting_end = 177
    plotting_step = 0.1

    # ? Setup specifications
    x_range = (0, 100)
    y_range = (0, 40)
    scenario_start = 0
    scenario_end = 200
    start_coords = (2, 2)
    start_time = 0
    goal_coords = (98, 36)
    min_radius = 2
    max_radius = 8
    stepsize = 0.1

    samples = 100
    pruning = 0
    obstacles = []

    ### Create static images of scenery in scenar_setup
    # create geometric obstacles
    poly1 = ShapelyPolygon([(50, 8), (60, 8), (60, 32), (50, 32)])
    poly2 = ShapelyPolygon([(15, 0), (30, 0), (30, 20), (15, 20)])
    poly3 = ShapelyPolygon([(75, 20), (95, 20), (95, 40), (75, 40)])

    if generate_setup_images:
        obs1_demo = Polygon(geometry=poly1)
        obs2_demo = Polygon(
            geometry=poly2, time_interval=Interval(10, 100, closed="both")
        )
        obs3_demo = Polygon(
            geometry=poly3, time_interval=Interval(10, 100, closed="both")
        )
        obs_demo = [obs1_demo, obs2_demo, obs3_demo]

        # create environment
        env_demo = Environment()
        env_demo.add_obstacles(obs_demo)

        # create environment instance
        env_inst_demo = EnvironmentInstance(
            environment=env_demo,
            query_interval=Interval(
                scenario_start,
                scenario_end,
                closed="both",
            ),
            scenario_range_x=x_range,
            scenario_range_y=y_range,
            quiet=True,
        )

        ## 1) Plot the scenario with all obstacle visible
        fig = plt.figure(figsize=(6, 2.4))
        env_inst_demo.plot(fig=fig)
        fig.tight_layout()
        plt.savefig("animations/scenario_setup/initial.png", dpi=300)

        ## 2) Plot the scenario with only static obstacle active
        fig = plt.figure(figsize=(6, 2.4))
        env_inst_demo.plot(fig=fig, query_time=0, show_inactive=True)
        fig.tight_layout()
        plt.savefig("animations/scenario_setup/static.png", dpi=300)

        ## 3) Plot the scenario with only dynamic obstacle active
        env_demo.reset()
        obs1_dyn = Polygon(
            geometry=poly1, time_interval=Interval(150, 1000, closed="both")
        )
        env_demo.add_obstacles([obs1_dyn, obs2_demo, obs3_demo])
        env_inst_demo = EnvironmentInstance(
            environment=env_demo,
            query_interval=Interval(
                scenario_start,
                scenario_end,
                closed="both",
            ),
            scenario_range_x=x_range,
            scenario_range_y=y_range,
            quiet=True,
        )
        fig = plt.figure(figsize=(6, 2.4))
        env_inst_demo.plot(fig=fig, query_time=20, show_inactive=True)
        fig.tight_layout()
        plt.savefig("animations/scenario_setup/dynamic.png", dpi=300)

        ## 4) Highlight task without obstacle active
        fig = plt.figure(figsize=(6, 2.4))
        env_inst_demo.plot(fig=fig, query_time=120, show_inactive=True)
        plt.plot(start_coords[0], start_coords[1], "ro")
        plt.plot(goal_coords[0], goal_coords[1], "go")
        fig.tight_layout()
        plt.savefig("animations/scenario_setup/task.png", dpi=300)

    ### Create video of scenario in animations (with pictures in scenar_images)
    if generate_images:
        # create static obstacles
        obs1 = Polygon(geometry=poly1)
        obstacles = obstacles + [obs1]

        # create dynamic obstacles
        obs2 = Polygon(geometry=poly2, time_interval=Interval(0, 10, closed="both"))
        obstacles = obstacles + [obs2]
        obs3 = Polygon(geometry=poly3, time_interval=Interval(40, 200, closed="both"))
        obstacles = obstacles + [obs3]

        # create random environment
        env_obj = Environment()
        env_obj.add_obstacles(obstacles)

        # create environment instance
        env = EnvironmentInstance(
            environment=env_obj,
            query_interval=Interval(
                scenario_start,
                scenario_end,
                closed="both",
            ),
            scenario_range_x=x_range,
            scenario_range_y=y_range,
            quiet=True,
        )

        ## Run TA-PRM algorithms

        # prepare the TA-PRM graph
        graph = Graph(
            num_samples=samples,
            env=env,
            seed=seed,
            quiet=True,
        )

        # connect start and goal node
        graph.connect_start(coords=start_coords)
        graph.connect_goal(coords=goal_coords, quiet=True)
        ta_prm = TAPRM(graph=graph)

        # plan a path with TA-PRM (including temporal pruning)
        start = time.time()
        success_taprm, path_taprm, _, _ = ta_prm.plan_temporal(
            start_time=start_time, temporal_precision=pruning, quiet=True
        )
        runtime_taprm = time.time() - start
        cost_taprm = graph.path_cost(path_taprm)
        print("Runtime TA-PRM (pruning): ", runtime_taprm)
        print("Path:", path_taprm)
        print("Cost:", cost_taprm)

        ## run RRT and RRT* with dynamic obstacle replanning
        replanner = ReplanningRRT(env=env, seed=seed)

        start = time.time()
        path_rrt, runs_rrt, prev_paths_rrt = replanner.run(
            samples=samples,
            stepsize=stepsize,
            start=start_coords,
            goal=goal_coords,
            query_time=start_time,
            rewiring=False,
            prev_path=[ShapelyPoint(*start_coords)],
            dynamic_obstacles=True,
            quiet=True,
        )
        runtime_rrt = time.time() - start
        cost_rrt = replanner.get_path_cost(sol_path=path_rrt)
        print("Runtime RRT: ", runtime_rrt)
        print("Path:", path_rrt)
        print("Cost:", cost_rrt)

        start = time.time()
        path_rrt_star, runs_rrt_star, prev_paths_rrt_star = replanner.run(
            samples=samples,
            stepsize=stepsize,
            start=start_coords,
            goal=goal_coords,
            query_time=start_time,
            rewiring=True,
            prev_path=[ShapelyPoint(*start_coords)],
            dynamic_obstacles=True,
            quiet=True,
        )
        runtime_rrt_star = time.time() - start
        cost_rrt_star = replanner.get_path_cost(sol_path=path_rrt_star)
        print("Runtime RRT*: ", runtime_rrt_star)
        print("Path:", path_rrt_star)
        print("Cost:", cost_rrt_star)

        ## Plot all algorithm results in the same figure
        # pre-compute the timed paths for TA-PRM* and TA-PRM
        timed_path_taprm = get_timed_path(
            graph=graph, sol_path=path_taprm, start_time=start_time
        )
        timed_path_rrt = get_timed_path_rrt(sol_path=path_rrt, start_time=start_time)
        timed_path_rrt_star = get_timed_path_rrt(
            sol_path=path_rrt_star, start_time=start_time
        )

        # ensure that both RRT and RRT* used one replanning for plotting to work as expected
        assert len(prev_paths_rrt) == 1
        assert len(prev_paths_rrt_star) == 1

        # iterate from plotting_start to plotting_end with plotting_step
        for plotting_time in np.arange(plotting_start, plotting_end, plotting_step):
            plotting_time = round(plotting_time, 1)
            print("Saving figure at time: ", plotting_time)
            fig = plt.figure(figsize=(6, 2.4))
            env.plot(fig=fig, query_time=plotting_time, show_inactive=True)

            # plot TA-PRM path and current position
            plot_taprm_path(
                sol_path=path_taprm, graph=graph, color="blue", label="TA-PRM* / TA-PRM"
            )
            if plotting_time <= cost_taprm:
                curr_pos_taprm = get_current_pos_timed_path(
                    time=plotting_time, timed_path=timed_path_taprm, graph=graph
                )

            # plot RRT path
            if plotting_time < prev_paths_rrt[0][1]:
                plot_rrt_path(sol_path=prev_paths_rrt[0][0], color="black", label="RRT")
            else:
                plot_rrt_path(sol_path=path_rrt, color="black", label="RRT")

            if plotting_time < cost_rrt:
                curr_pos_rrt = get_current_pos_timed_path(
                    time=plotting_time, timed_path=timed_path_rrt, graph=graph
                )

            # plot RRT* path
            if plotting_time < prev_paths_rrt_star[0][1]:
                plot_rrt_path(
                    sol_path=prev_paths_rrt_star[0][0], color="orange", label="RRT*"
                )
            else:
                plot_rrt_path(
                    sol_path=path_rrt_star,
                    color="orange",
                    label="RRT*",
                )

            if plotting_time < cost_rrt_star:
                curr_pos_rrt_star = get_current_pos_timed_path(
                    time=plotting_time,
                    timed_path=timed_path_rrt_star,
                    graph=graph,
                )

            # plot current positions on all algorithm paths
            if plotting_time <= cost_taprm:
                plt.plot(
                    curr_pos_taprm[0],
                    curr_pos_taprm[1],
                    color="red",
                    marker="o",
                    markersize=6,
                )

            if plotting_time <= cost_rrt:
                plt.plot(
                    curr_pos_rrt[0],
                    curr_pos_rrt[1],
                    color="red",
                    marker="o",
                    markersize=6,
                )

            if plotting_time <= cost_rrt_star:
                plt.plot(
                    curr_pos_rrt_star[0],
                    curr_pos_rrt_star[1],
                    color="red",
                    marker="o",
                    markersize=6,
                )

            # add start and goal markings to plot
            plt.plot(start_coords[0], start_coords[1], "go", markersize=8)
            plt.plot(goal_coords[0], goal_coords[1], "go", markersize=8)

            # add time to figure
            plt.text(2, 35, "$t = {}$".format(plotting_time), fontsize=10)

            # add legend to the plot
            plt.legend(
                bbox_to_anchor=(0, 1.02, 1, 0.2),
                loc="lower left",
                mode="expand",
                borderaxespad=0,
                ncol=4,
                fontsize=10,
            )
            fig.tight_layout()
            plt.savefig(
                f"animations/scenario_images/scenario_{plotting_time}.png", dpi=300
            )

            plt.savefig(
                f"animations/scenario_images/scenario_{plotting_time}_high_res.png",
                dpi=400,
            )

            plt.savefig(
                f"animations/scenario_images/scenario_{plotting_time}_highest_res.png",
                dpi=500,
            )

            # close the plot
            plt.close()

    ## Create videos from images
    if generate_videos:
        create_video(
            filename="animations/scenario_videos/slow",
            tmin=0.0,
            tmax=177,
            step=0.3,
            fps=30,
        )
        create_video(
            filename="animations/scenario_videos/medium",
            tmin=0.0,
            tmax=177,
            step=0.5,
            fps=30,
        )
        create_video(
            filename="animations/scenario_videos/fast",
            tmin=0.0,
            tmax=177,
            step=1,
            fps=30,
        )
        create_video(
            filename="animations/scenario_videos/slow_high_fps",
            tmin=0.0,
            tmax=177,
            step=0.1,
            fps=90,
        )
        create_video(
            filename="animations/scenario_videos/medium_high_fps",
            tmin=0.0,
            tmax=177,
            step=0.2,
            fps=50,
        )
        create_video(
            filename="animations/scenario_videos/fast_high_fps",
            tmin=0.0,
            tmax=177,
            step=0.3,
            fps=100,
        )

        create_video(
            filename="animations/scenario_videos/high_res_slow",
            tmin=0.0,
            tmax=177,
            step=0.3,
            fps=30,
        )
        create_video(
            filename="animations/scenario_videos/high_res_medium",
            tmin=0.0,
            tmax=177,
            step=0.5,
            fps=30,
        )
        create_video(
            filename="animations/scenario_videos/high_res_fast",
            tmin=0.0,
            tmax=177,
            step=1,
            fps=30,
        )
        create_video(
            filename="animations/scenario_videos/high_res_slow_high_fps",
            tmin=0.0,
            tmax=177,
            step=0.1,
            fps=90,
        )
        create_video(
            filename="animations/scenario_videos/high_res_medium_high_fps",
            tmin=0.0,
            tmax=177,
            step=0.2,
            fps=50,
        )
        create_video(
            filename="animations/scenario_videos/high_res_fast_high_fps",
            tmin=0.0,
            tmax=177,
            step=0.3,
            fps=100,
        )

        create_video(
            filename="animations/scenario_videos/highest_res_slow",
            tmin=0.0,
            tmax=177,
            step=0.3,
            fps=30,
        )
        create_video(
            filename="animations/scenario_videos/highest_res_medium",
            tmin=0.0,
            tmax=177,
            step=0.5,
            fps=30,
        )
        create_video(
            filename="animations/scenario_videos/highest_res_fast",
            tmin=0.0,
            tmax=177,
            step=1,
            fps=30,
        )
        create_video(
            filename="animations/scenario_videos/highest_res_slow_high_fps",
            tmin=0.0,
            tmax=177,
            step=0.1,
            fps=90,
        )
        create_video(
            filename="animations/scenario_videos/highest_res_medium_high_fps",
            tmin=0.0,
            tmax=177,
            step=0.2,
            fps=50,
        )
        create_video(
            filename="animations/scenario_videos/highest_res_fast_high_fps",
            tmin=0.0,
            tmax=177,
            step=0.3,
            fps=100,
        )
