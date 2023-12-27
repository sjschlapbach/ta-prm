from typing import List, Union
from shapely.geometry import (
    Polygon as ShapelyPolygon,
    Point as ShapelyPoint,
    LineString as ShapelyLine,
)

from .environment import Environment

# ! idea: this class stores similar information as before, but
# only an obstacle list (no recurrence) and selects the correct
# ones based on the query interval (all which interesect with the query interval).

# ! additionally a spatial index is created for the obstacles,
# ! so that the closest obstacle can be found efficiently


class EnvironmentInstance:
    # TODO: Docstring

    def __init__(self, environment: Environment):
        pass

    # TODO - creation method for environment instance, creating a spatial index (which also works for lines) and storing the obstacles in an indexed list

    # TODO - update to new obstacles structure and new spatial index
    def closest_polygon_distance(self, point: ShapelyPoint) -> float:
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

    # TODO - update to new obstacles structure and new spatial index
    def closest_line_distance(self, line: ShapelyLine) -> float:
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
