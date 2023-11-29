import pytest
from shapely.geometry import Point as ShapelyPoint, LineString, Polygon
from pandas import Interval
from enum import Enum
from src.obstacles.point import Point
from matplotlib import pyplot as plt


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

    def test_temporal_collision_check(self):
        in1 = Interval(0, 10, closed="both")
        in2 = Interval(5, 15, closed="both")
        in3 = Interval(15, 25, closed="both")

        assert in1.overlaps(in2) == True
        assert in1.overlaps(in3) == False
        assert in2.overlaps(in1) == True
        assert in2.overlaps(in3) == True
        assert in3.overlaps(in1) == False
        assert in3.overlaps(in2) == True

        point = self.setup_method()
        test_pt = ShapelyPoint(0.5, 0.5)

        # collision check with point inside spatial area
        point.set_interval(0, 10)
        assert point.check_collision(test_pt, query_time=5) == True
        assert point.check_collision(test_pt, query_interval=in1) == True
        assert point.check_collision(test_pt, query_time=15) == False
        assert point.check_collision(test_pt, query_interval=in3) == False

        # collision check with point outside spatial area
        test_pt = ShapelyPoint(5.5, 5.5)
        assert point.check_collision(test_pt, query_time=5) == False
        assert point.check_collision(test_pt, query_interval=in1) == False
        assert point.check_collision(test_pt, query_time=15) == False
        assert point.check_collision(test_pt, query_interval=in3) == False

        # collision check with point on the edge of spatial area
        sqrt2_inv = 1 / 2**0.5
        test_pt = ShapelyPoint(sqrt2_inv, sqrt2_inv)
        assert point.check_collision(test_pt, query_time=5) == True
        assert point.check_collision(test_pt, query_interval=in1) == True
        assert point.check_collision(test_pt, query_time=15) == False
        assert point.check_collision(test_pt, query_interval=in3) == False

        # collision check with line and different temporal queries
        line = LineString([(0, 0), (1, 1)])
        assert point.check_collision(line, query_time=5) == True
        assert point.check_collision(line, query_interval=in1) == True
        assert point.check_collision(line, query_time=15) == False
        assert point.check_collision(line, query_interval=in3) == False

        # collision check with polygon and different temporal queries
        polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        assert point.check_collision(polygon, query_time=5) == True
        assert point.check_collision(polygon, query_interval=in1) == True
        assert point.check_collision(polygon, query_time=15) == False
        assert point.check_collision(polygon, query_interval=in3) == False

    def test_plot(self):
        fig = plt.figure()

        # Test case 1: No query time or query interval provided
        point = Point(geometry=ShapelyPoint(0, 0), radius=5)
        point.plot(fig=fig)  # Plot the point and circle

        # Test case 2: Query time is provided, but point has no time interval
        point = Point(geometry=ShapelyPoint(2, 2), radius=5)
        point.plot(query_time=5, fig=fig)

        # Test case 3: Query time is within the point's time interval
        point = Point(
            geometry=ShapelyPoint(-1, 3),
            radius=5,
            time_interval=Interval(0, 10, closed="both"),
        )
        point.plot(query_time=5, fig=fig)  # Plot the point and circle at query time 5

        # Test case 4: Query interval overlaps with the point's time interval
        point = Point(
            geometry=ShapelyPoint(-2, 0),
            radius=5,
            time_interval=Interval(0, 10, closed="both"),
        )
        point.plot(query_interval=Interval(5, 15), fig=fig)

        # Test case 5: Query time is outside the point's time interval
        point = Point(
            geometry=ShapelyPoint(0, -2),
            radius=5,
            time_interval=Interval(0, 10, closed="both"),
        )
        point.plot(query_time=15, fig=fig)

        # Test case 6: Query interval does not overlap with the point's time interval
        point = Point(
            geometry=ShapelyPoint(2, -2),
            radius=5,
            time_interval=Interval(0, 10, closed="both"),
        )
        point.plot(query_interval=Interval(15, 25), fig=fig)

        # Test case 7: No figure provided
        point = Point(geometry=ShapelyPoint(0, 0), radius=5)
        point.plot()  # Plot the point and circle on a new figure
