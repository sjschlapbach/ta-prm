from typing import List
from shapely.geometry import Polygon, Point, LineString
from shapely import wkt
import json
import matplotlib.pyplot as plt


class Environment:
    """
    A class to represent an environment consisting of a list of shapely polygon objects.

    ...

    Attributes
    ----------
    polygons : List[Polygon]
        a list of shapely polygon objects representing the environment

    Methods
    -------
    plot()
        Plots the polygons in the environment using matplotlib.

    closest_polygon_distance(point: Point) -> float
        Computes the distance between a shapely point object and the closest polygon in the environment.

    closest_line_distance(line: LineString) -> float
        Computes the distance between a shapely line object and the closest polygon in the environment.

    change_polygons(new_polygons: List[Polygon])
        Changes the polygons stored in the environment.

    save(filepath: str)
        Logs the polygons stored in the environment to a file in JSON format.

    load(filepath: str)
        Loads the polygons stored in a file into the environment.
    """

    def __init__(self, polygons: List[Polygon] = None, filepath: str = None):
        """
        Parameters
        ----------
        polygons : List[Polygon], optional
            a list of shapely polygon objects representing the environment
        filepath : str, optional
            the path to the file where the polygons are stored
        """
        if filepath is not None:
            self.load(filepath)
        elif polygons is not None:
            self.polygons = polygons
        else:
            self.polygons = []

    def plot(self):
        """
        Plots the polygons in the environment using matplotlib.
        """
        for polygon in self.polygons:
            x, y = polygon.exterior.xy
            plt.fill(x, y, color="#0000ff", alpha=0.5)
        plt.show()

    def closest_polygon_distance(self, point: Point) -> float:
        """
        Computes the distance between a shapely point object and the closest polygon in the environment.
        If there is a collision between the point and a polygon, the distance returned will be 0.0.

        Parameters
        ----------
        point : Point
            a shapely point object

        Returns
        -------
        float
            the distance between the point and the closest polygon in the environment, or 0.0 if there is a collision
        """
        return min([polygon.distance(point) for polygon in self.polygons])

    def closest_line_distance(self, line: LineString) -> float:
        """
        Computes the distance between a shapely line object and the closest polygon in the environment.

        Parameters
        ----------
        line : LineString
            a shapely line object

        Returns
        -------
        float
            the distance between the line and the closest polygon in the environment
        """
        return min([line.distance(polygon) for polygon in self.polygons])

    def change_polygons(self, new_polygons: List[Polygon]):
        """
        Changes the polygons stored in the environment.

        Parameters
        ----------
        new_polygons : List[Polygon]
            a list of shapely polygon objects representing the new environment
        """
        self.polygons = new_polygons

    def save(self, filepath: str):
        """
        Logs the polygons stored in the environment to a file in JSON format.

        Parameters
        ----------
        filepath : str
            the path to the file where the polygons will be logged
        """
        with open(filepath, 'w') as f:
            json.dump([polygon.wkt for polygon in self.polygons], f)

    def load(self, filepath: str):
        """
        Loads the polygons stored in a file into the environment.

        Parameters
        ----------
        filepath : str
            the path to the file where the polygons are stored
        """
        with open(filepath, 'r') as f:
            polygons_wkt = json.load(f)
        self.polygons = [wkt.loads(element) for element in polygons_wkt]
