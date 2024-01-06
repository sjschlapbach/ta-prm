from src.obstacles.line import Line
from src.obstacles.point import Point
from src.obstacles.polygon import Polygon
from src.envs.environment import Environment
from src.util.recurrence import Recurrence

from pandas import Interval
import matplotlib.pyplot as plt
from shapely.geometry import (
    Point as ShapelyPoint,
    LineString as ShapelyLine,
    Polygon as ShapelyPolygon,
)


def plot_environment(plotting: bool = True):
    # initialize point objects
    sh_pt1 = ShapelyPoint(0, 0)
    sh_pt2 = ShapelyPoint(8, 3)
    pt1 = Point(sh_pt1)
    pt2 = Point(sh_pt2, Interval(5, 20), recurrence=Recurrence.MINUTELY)

    # initialize line objects
    sh_line1 = ShapelyLine([(2, 2), (1, 3)])
    sh_line2 = ShapelyLine([(5, 4), (2, 5)])
    line1 = Line(sh_line1)
    line2 = Line(sh_line2, Interval(10, 40), radius=0.5, recurrence=Recurrence.MINUTELY)

    # initialize polygon objects
    sh_poly1 = ShapelyPolygon([(6, 6), (7, 7), (6, 8)])
    sh_poly2 = ShapelyPolygon([(9, 9), (10, 10), (11, 11)])
    poly1 = Polygon(sh_poly1)
    poly2 = Polygon(
        sh_poly2, Interval(15, 40), radius=1.5, recurrence=Recurrence.HOURLY
    )

    # create an environment with all types of obstacles
    env = Environment(obstacles=[pt1, line1, poly1, pt2, line2, poly2])

    # initialize a figure
    fig = plt.figure(figsize=(8, 8))

    # run the query time from 1 to 30 in increments of 1 and plot the corresponding result and write the current time to the figure
    for query_time in range(1, 90):
        env.plot(query_time=query_time, fig=fig)
        plt.title(f"Query Time: {query_time}")
        plt.xlim([-1, 15])
        plt.ylim([-1, 15])

        if plotting:
            plt.draw()
            plt.pause(0.2)
            plt.clf()


if __name__ == "__main__":
    plot_environment()
