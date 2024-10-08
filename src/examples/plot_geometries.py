from src.obstacles.line import Line
from src.obstacles.point import Point
from src.obstacles.polygon import Polygon

from shapely.geometry import (
    Point as ShapelyPoint,
    LineString as ShapelyLine,
    Polygon as ShapelyPolygon,
)
from pandas import Interval
import matplotlib.pyplot as plt


def plot_geometries(plotting: bool = True, query_time: float = None):
    if query_time is None:
        # get a query time from the user through a command line argument
        query_time = float(input("Enter a query time: "))

    # create a point and a line
    pt1 = Point(
        geometry=ShapelyPoint(-3, 0),
        time_interval=Interval(0, 10, closed="both"),
        radius=0.5,
    )
    ln1 = Line(
        geometry=ShapelyLine([(-2, -2), (-1, -1)]),
        time_interval=Interval(0, 10, closed="both"),
        radius=1.0,
    )
    pg1 = Polygon(
        geometry=ShapelyPolygon([(1, 0), (1, 1), (2, 1)]),
        time_interval=Interval(0, 10, closed="both"),
        radius=0.5,
    )

    # create a point and line without temporal constraints and no padding
    pt2 = Point(geometry=ShapelyPoint(2, 2))
    ln2 = Line(geometry=ShapelyLine([(2, -1), (3, 1), (4, 3)]))
    pg2 = Polygon(geometry=ShapelyPolygon([(6, 0), (7, 1), (8, 0)]))

    # create a point and line without temporal constraints padding
    pt3 = Point(geometry=ShapelyPoint(1, 7), radius=1.0)
    ln3 = Line(geometry=ShapelyLine([(4, 7), (7, 6), (7, 3)]), radius=0.5)
    pg3 = Polygon(geometry=ShapelyPolygon([(5, -2), (6, -3), (7, -1)]), radius=0.5)

    # create a figure from (-5, -5) to (10, 10)
    fig = plt.figure(figsize=(8, 8))
    pt1.plot(query_time=query_time, fig=fig)
    ln1.plot(query_time=query_time, fig=fig)
    pg1.plot(query_time=query_time, fig=fig)
    pt2.plot(query_time=query_time, fig=fig)
    ln2.plot(query_time=query_time, fig=fig)
    pg2.plot(query_time=query_time, fig=fig)
    pt3.plot(query_time=query_time, fig=fig)
    ln3.plot(query_time=query_time, fig=fig)
    pg3.plot(query_time=query_time, fig=fig)

    plt.xlim([-5, 10])
    plt.ylim([-5, 10])

    if plotting:
        plt.show()


if __name__ == "__main__":
    plot_geometries()
