import pytest
from shapely.geometry import Point as ShapelyPoint, LineString, Polygon
from pandas import Interval
from enum import Enum
from src.geometry.point import Point


class TestPoint:
    def setup_method(self):
        point = Point(ShapelyPoint(0, 0), Interval(0, 10, closed="both"), 1.0)
        return point

    def test_setup(self):
        # Test constructor with out anything
        point = Point()
        assert point.geometry == None
        assert point.time_interval == None
        assert point.radius == 0

        # Test constructor with geometry
        point = Point(ShapelyPoint(0, 0))
        assert point.geometry == ShapelyPoint(0, 0)
        assert point.time_interval == None
        assert point.radius == 0

        # Test constructor with geometry and time interval
        point = Point(ShapelyPoint(0, 0), Interval(0, 10, closed="both"))
        assert point.geometry == ShapelyPoint(0, 0)
        assert point.time_interval == Interval(0, 10, closed="both")
        assert point.radius == 0

        # Test constructor with geometry, time interval, and radius
        point = Point(ShapelyPoint(0, 0), Interval(0, 10, closed="both"), 1.0)
        assert point.geometry == ShapelyPoint(0, 0)
        assert point.time_interval == Interval(0, 10, closed="both")
        assert point.radius == 1.0

    def test_set_geometry(self):
        point = self.setup_method()
        point.set_geometry(1, 2)
        assert point.geometry == ShapelyPoint(1, 2)

    def test_set_interval(self):
        point = self.setup_method()
        point.set_interval(5, 15)
        assert point.time_interval == Interval(5, 15, closed="both")

    def test_set_radius(self):
        point = self.setup_method()
        point.set_radius(2.0)
        assert point.radius == 2.0

    def test_check_collision_with_point(self):
        point = self.setup_method()

        # collision check with point outside
        other_point = ShapelyPoint(1, 1)
        assert point.check_collision(other_point) == False

        # collision check with point inside
        other_point = ShapelyPoint(0, 0)
        assert point.check_collision(other_point) == True

        # collision check with point on the edge
        sqrt2_inv = 1 / 2**0.5
        other_point = ShapelyPoint(sqrt2_inv, sqrt2_inv)
        assert point.check_collision(other_point) == True

        # collision check with point slightly outside the area
        other_point = ShapelyPoint(sqrt2_inv + 10e-5, sqrt2_inv + 10e-5)
        assert point.check_collision(other_point) == False

    def test_check_collision_with_line_string(self):
        point = self.setup_method()

        # collision check with line outside
        line = LineString([(1, 1), (2, 2)])
        assert point.check_collision(line) == False

        # collision check with line passing through
        line = LineString([(0, 0), (0, 2), (2, 2)])
        assert point.check_collision(line) == True

        # collision check with line on the edge
        line = LineString([(1, -2), (1, 2)])
        assert point.check_collision(line) == True

        # collision check with line slightly outside the area
        line = LineString([(1 + 10e-5, -2), (1 + 10e-5, 2)])
        assert point.check_collision(line) == False

    def test_check_collision_with_polygon(self):
        point = self.setup_method()

        # collision check with polygon outside
        polygon = Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])
        assert point.check_collision(polygon) == False

        # collision check with polygon containing point
        polygon = Polygon([(0, 0), (0, 2), (2, 2), (2, 0)])
        assert point.check_collision(polygon) == True

        # collision check with polygon on the edge
        polygon = Polygon([(1, -2), (1, 2), (2, 2)])
        assert point.check_collision(polygon) == True

        # collision check with polygon slightly outside the area
        polygon = Polygon([(1 + 10e-5, -2), (1 + 10e-5, 2), (2, 2)])
        assert point.check_collision(polygon) == False

    def test_check_collision_with_invalid_shape(self):
        point = self.setup_method()
        with pytest.raises(ValueError):
            point.check_collision("invalid shape")
