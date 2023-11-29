import pytest
from shapely.geometry import Point, LineString, Polygon
from pandas import Interval
from src.obstacles.line import Line
from matplotlib import pyplot as plt


class TestLine:
    def setup_method(self):
        line = Line(LineString([(0, 0), (1, 1)]), Interval(0, 10, closed="both"), 1.0)
        return line

    def test_setup(self):
        # Test constructor with out anything
        line = Line()
        assert line.geometry == None
        assert line.time_interval == None
        assert line.radius == 0

        # Test constructor with start and end points
        line = Line(LineString([(0, 0), (1, 1)]))
        assert line.geometry == LineString([(0, 0), (1, 1)])
        assert line.time_interval == None
        assert line.radius == 0

        # Test constructor with start and end points, time interval, and radius
        line = Line(LineString([(0, 0), (1, 1)]), Interval(0, 10, closed="both"), 1.0)
        assert line.geometry == LineString([(0, 0), (1, 1)])
        assert line.time_interval == Interval(0, 10, closed="both")
        assert line.radius == 1.0

    def test_set_geometry(self):
        line = self.setup_method()
        line.set_geometry([(1, 2), (1, 1)])
        assert line.geometry == LineString([(1, 2), (1, 1)])

    def test_set_interval(self):
        line = self.setup_method()
        line.set_interval(5, 15)
        assert line.time_interval == Interval(5, 15, closed="both")

    def test_set_radius(self):
        line = self.setup_method()
        line.set_radius(2.0)
        assert line.radius == 2.0

    def test_check_collision_with_point(self):
        line = self.setup_method()

        # collision check with point outside
        point = Point(2, 2)
        assert line.check_collision(point) == False

        # collision check with point inside
        point = Point(0.5, 0.5)
        assert line.check_collision(point) == True

        # collision check with point on the edge
        point = Point(1, 1)
        assert line.check_collision(point) == True

        # collision check with point inside radius region
        point = Point(1.5, 1.5)
        assert line.check_collision(point) == True

        # collision check with point slightly outside the line
        point = Point(1.75, 1.75)
        assert line.check_collision(point) == False

    def test_check_collision_with_line_string(self):
        line = self.setup_method()

        # collision check with line outside
        other_line = LineString([(2, 2), (3, 3)])
        assert line.check_collision(other_line) == False

        # collision check with line passing through
        other_line = LineString([(0, 0), (1, 1), (2, 2)])
        assert line.check_collision(other_line) == True

        # collision check with line on the edge
        other_line = LineString([(0.5, 0.5), (1.5, 1.5)])
        assert line.check_collision(other_line) == True

        # collision check with line inside radius region
        other_line = LineString([(1.5, 1.5), (2.5, 2.5)])
        assert line.check_collision(other_line) == True

        # collision check with line slightly outside the line
        other_line = LineString([(1.75, 1.75), (2.5, 2.5)])
        assert line.check_collision(other_line) == False

    def test_check_collision_with_polygon(self):
        line = self.setup_method()

        # collision check with polygon outside
        polygon = Polygon([(2, 2), (2, 3), (3, 3), (3, 2)])
        assert line.check_collision(polygon) == False

        # collision check with polygon containing line
        polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        assert line.check_collision(polygon) == True

        # collision check with polygon on the edge
        polygon = Polygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)])
        assert line.check_collision(polygon) == True

        # collision check with polygon inside radius region
        polygon = Polygon([(1.5, 1.5), (1.5, 2.5), (2.5, 2.5), (2.5, 1.5)])
        assert line.check_collision(polygon) == True

        # collision check with polygon slightly outside the line
        polygon = Polygon([(1.75, 1.75), (1.75, 2.5), (2.5, 2.5), (2.5, 1.75)])
        assert line.check_collision(polygon) == False

    def test_check_collision_with_invalid_shape(self):
        line = self.setup_method()
        with pytest.raises(ValueError):
            line.check_collision("invalid shape")

    def test_temporal_collision_check(self):
        in1 = Interval(0, 10, closed="both")
        in2 = Interval(5, 15, closed="both")
        in3 = Interval(15, 25, closed="both")

        line = self.setup_method()
        test_line = LineString([(0.5, 0.5), (1.5, 1.5)])

        # collision check with line inside spatial area
        line.set_interval(0, 10)
        assert line.is_active(query_time=5) == True
        assert line.check_collision(test_line, query_time=5) == True
        assert line.is_active(query_interval=in1) == True
        assert line.check_collision(test_line, query_interval=in1) == True
        assert line.is_active(query_time=15) == False
        assert line.check_collision(test_line, query_time=15) == False
        assert line.is_active(query_interval=in3) == False
        assert line.check_collision(test_line, query_interval=in3) == False

        # collision check with line outside spatial area
        test_line = LineString([(5.5, 5.5), (6.5, 6.5)])
        assert line.check_collision(test_line, query_time=5) == False
        assert line.check_collision(test_line, query_interval=in1) == False
        assert line.check_collision(test_line, query_time=15) == False
        assert line.check_collision(test_line, query_interval=in3) == False

        # collision check with line on the edge of spatial area
        test_line = LineString([(1, 1), (2, 2)])
        assert line.check_collision(test_line, query_time=5) == True
        assert line.check_collision(test_line, query_interval=in1) == True
        assert line.check_collision(test_line, query_time=15) == False
        assert line.check_collision(test_line, query_interval=in3) == False

        # collision check with point and different temporal queries
        point = Point(0.5, 0.5)
        assert line.check_collision(point, query_time=5) == True
        assert line.check_collision(point, query_interval=in1) == True
        assert line.check_collision(point, query_time=15) == False
        assert line.check_collision(point, query_interval=in3) == False

        # collision check with polygon and different temporal queries
        polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        assert line.check_collision(polygon, query_time=5) == True
        assert line.check_collision(polygon, query_interval=in1) == True
        assert line.check_collision(polygon, query_time=15) == False
        assert line.check_collision(polygon, query_interval=in3) == False

    def test_check_collision_without_time_interval(self):
        line = Line(LineString([(0, 0), (1, 1)]), None, 1.0)

        # collision check with point outside
        point = Point(2, 2)
        assert line.check_collision(point) == False

        # collision check with point inside
        point = Point(0.5, 0.5)
        assert line.check_collision(point) == True

        # collision check with point inside at arbitrary time
        point = Point(0.5, 0.5)
        assert line.check_collision(point, query_time=5) == True

        # collision check with point outside at arbitrary time
        point = Point(2, 2)
        assert line.check_collision(point, query_time=5) == False

        # collision check with line inside
        other_line = LineString([(0, 0), (1, 1), (2, 2)])
        assert line.check_collision(other_line) == True

        # collision check with line outside
        other_line = LineString([(2, 2), (3, 3)])
        assert line.check_collision(other_line) == False

        # collision check with line inside at arbitrary time
        other_line = LineString([(0, 0), (1, 1), (2, 2)])
        assert line.check_collision(other_line, query_time=5) == True

        # collision check with line outside at arbitrary time
        other_line = LineString([(2, 2), (3, 3)])
        assert line.check_collision(other_line, query_time=5) == False

        # collision check with polygon inside
        polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        assert line.check_collision(polygon) == True

        # collision check with polygon outside
        polygon = Polygon([(2, 2), (2, 3), (3, 3), (3, 2)])
        assert line.check_collision(polygon) == False

        # collision check with polygon inside at arbitrary time
        polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        assert line.check_collision(polygon, query_time=5) == True

        # collision check with polygon outside at arbitrary time
        polygon = Polygon([(2, 2), (2, 3), (3, 3), (3, 2)])
        assert line.check_collision(polygon, query_time=5) == False

    def test_plot(self):
        fig = plt.figure()

        # Test case 1: No query time or query interval provided
        line = Line(LineString([(0, 0), (1, 1)]), Interval(0, 10, closed="both"), 1.0)
        line.plot(fig=fig)  # Plot the line

        # Test case 2: Query time is provided, but line has no time interval
        line = Line(LineString([(2, 2), (3, 3)]))
        line.plot(query_time=5, fig=fig)

        # Test case 3: Query time is within the line's time interval
        line = Line(LineString([(0, 0), (1, 1)]), Interval(0, 10, closed="both"), 1.0)
        line.plot(query_time=5, fig=fig)  # Plot the line at query time 5

        # Test case 4: Query interval overlaps with the line's time interval
        line = Line(LineString([(0, 0), (1, 1)]), Interval(0, 10, closed="both"), 1.0)
        line.plot(query_interval=Interval(5, 15), fig=fig)

        # Test case 5: Query time is outside the line's time interval
        line = Line(LineString([(0, 0), (1, 1)]), Interval(0, 10, closed="both"), 1.0)
        line.plot(query_time=15, fig=fig)

        # Test case 6: Query interval does not overlap with the line's time interval
        line = Line(LineString([(0, 0), (1, 1)]), Interval(0, 10, closed="both"), 1.0)
        line.plot(query_interval=Interval(15, 25), fig=fig)

        # Test case 7: No figure provided
        line = Line(LineString([(0, 0), (1, 1)]), Interval(0, 10, closed="both"), 1.0)
        line.plot()  # Plot the line on a new figure
