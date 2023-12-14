import pytest
from shapely.geometry import Point, LineString, Polygon as ShapelyPolygon
from pandas import Interval
from matplotlib import pyplot as plt

from src.obstacles.polygon import Polygon


class TestPolygon:
    def setup_method(self):
        polygon = Polygon(
            geometry=ShapelyPolygon([(0, 0), (1, 1), (1, 0)]),
            time_interval=Interval(0, 10),
            radius=1.0,
        )
        return polygon

    def test_setup(self):
        # Test constructor with out anything
        polygon = Polygon()
        assert polygon.geometry == None
        assert polygon.time_interval == None
        assert polygon.radius == 0

        # Test constructor with points
        polygon = Polygon(ShapelyPolygon([(0, 0), (1, 1), (1, 0)]))
        assert polygon.geometry == ShapelyPolygon([(0, 0), (1, 1), (1, 0)])

        # Test constructor with start and end points
        poly = Polygon(ShapelyPolygon([(0, 0), (1, 1), (1, 0)]), Interval(0, 10))
        assert poly.geometry == ShapelyPolygon([(0, 0), (1, 1), (1, 0)])
        assert poly.time_interval == Interval(0, 10)
        assert poly.radius == 0

        # Test constructor with start and end points, time interval, and radius
        poly = Polygon(
            ShapelyPolygon([(0, 0), (1, 1), (3, 3)]),
            Interval(0, 10, closed="both"),
            1.0,
        )
        assert poly.geometry == ShapelyPolygon([(0, 0), (1, 1), (3, 3)])
        assert poly.time_interval == Interval(0, 10, closed="both")
        assert poly.radius == 1.0

    def test_set_geometry(self):
        polygon = self.setup_method()
        polygon.set_geometry([(1, 2), (2, 2), (2, 1)])
        assert polygon.geometry == ShapelyPolygon([(1, 2), (2, 2), (2, 1)])

    def test_check_collision_with_point(self):
        polygon = self.setup_method()

        # collision check with point outside
        point = Point(2, 2)
        assert polygon.check_collision(point) == False

        # collision check with point inside
        point = Point(0.5, 0.5)
        assert polygon.check_collision(point) == True

        # collision check with point on the edge
        point = Point(1, 1)
        assert polygon.check_collision(point) == True

        # collision check with point inside radius region
        point = Point(1.5, 1.5)
        assert polygon.check_collision(point) == True

        # collision check with point slightly outside the polygon
        point = Point(1.75, 1.75)
        assert polygon.check_collision(point) == False

    def test_check_collision_with_line_string(self):
        polygon = self.setup_method()

        # collision check with line outside
        line = LineString([(2, 2), (3, 3)])
        assert polygon.check_collision(line) == False

        # collision check with line passing through
        line = LineString([(0, 0), (1, 1), (2, 2)])
        assert polygon.check_collision(line) == True

        # collision check with line on the edge
        line = LineString([(0.5, 0.5), (1.5, 1.5)])
        assert polygon.check_collision(line) == True

        # collision check with line inside radius region
        line = LineString([(1.5, 1.5), (2.5, 2.5)])
        assert polygon.check_collision(line) == True

        # collision check with line slightly outside the polygon
        line = LineString([(1.75, 1.75), (2.5, 2.5)])
        assert polygon.check_collision(line) == False

    def test_check_collision_with_polygon(self):
        polygon = self.setup_method()

        # collision check with polygon outside
        other_polygon = ShapelyPolygon([(2, 2), (2, 3), (3, 3), (3, 2)])
        assert polygon.check_collision(other_polygon) == False

        # collision check with polygon containing the polygon
        other_polygon = ShapelyPolygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        assert polygon.check_collision(other_polygon) == True

        # collision check with polygon on the edge
        other_polygon = ShapelyPolygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)])
        assert polygon.check_collision(other_polygon) == True

        # collision check with polygon inside radius region
        other_polygon = ShapelyPolygon([(1.5, 1.5), (1.5, 2.5), (2.5, 2.5), (2.5, 1.5)])
        assert polygon.check_collision(other_polygon) == True

        # collision check with polygon slightly outside the polygon
        other_polygon = ShapelyPolygon(
            [(1.75, 1.75), (1.75, 2.5), (2.5, 2.5), (2.5, 1.75)]
        )
        assert polygon.check_collision(other_polygon) == False

    def test_check_collision_with_invalid_shape(self):
        polygon = self.setup_method()
        with pytest.raises(ValueError):
            polygon.check_collision("invalid shape")

    def test_temporal_collision_check(self):
        in1 = Interval(0, 10, closed="both")
        in2 = Interval(5, 15, closed="both")
        in3 = Interval(15, 25, closed="both")

        polygon = self.setup_method()
        test_polygon = ShapelyPolygon([(0.5, 0.5), (1.5, 1.5), (2.5, 2.5)])

        # collision check with polygon inside spatial area
        polygon.set_interval(0, 10)
        assert polygon.is_active(query_time=5) == True
        assert polygon.check_collision(test_polygon, query_time=5) == True
        assert polygon.is_active(query_interval=in1) == True
        assert polygon.check_collision(test_polygon, query_interval=in1) == True
        assert polygon.is_active(query_time=15) == False
        assert polygon.check_collision(test_polygon, query_time=15) == False
        assert polygon.is_active(query_interval=in3) == False
        assert polygon.check_collision(test_polygon, query_interval=in3) == False

        # collision check with polygon outside spatial area
        test_polygon = ShapelyPolygon([(5.5, 5.5), (6.5, 6.5), (7.5, 7.5)])
        assert polygon.check_collision(test_polygon, query_time=5) == False
        assert polygon.check_collision(test_polygon, query_interval=in1) == False
        assert polygon.check_collision(test_polygon, query_time=15) == False
        assert polygon.check_collision(test_polygon, query_interval=in3) == False

        # collision check with polygon on the edge of spatial area
        test_polygon = ShapelyPolygon([(1, 1), (2, 2), (3, 3)])
        assert polygon.check_collision(test_polygon, query_time=5) == True
        assert polygon.check_collision(test_polygon, query_interval=in1) == True
        assert polygon.check_collision(test_polygon, query_time=15) == False
        assert polygon.check_collision(test_polygon, query_interval=in3) == False

        # collision check with point and different temporal queries
        point = Point(0.5, 0.5)
        assert polygon.check_collision(point, query_time=5) == True
        assert polygon.check_collision(point, query_interval=in1) == True
        assert polygon.check_collision(point, query_time=15) == False
        assert polygon.check_collision(point, query_interval=in3) == False

        # collision check with line and different temporal queries
        line = LineString([(0, 0), (1, 1), (2, 2)])
        assert polygon.check_collision(line, query_time=5) == True
        assert polygon.check_collision(line, query_interval=in1) == True
        assert polygon.check_collision(line, query_time=15) == False
        assert polygon.check_collision(line, query_interval=in3) == False

    def test_check_collision_without_time_interval(self):
        polygon = Polygon(ShapelyPolygon([(0, 0), (1, 1), (2, 2)]), None, 1.0)

        # collision check with point outside
        point = Point(3, 3)
        assert polygon.check_collision(point) == False

        # collision check with point inside
        point = Point(0.5, 0.5)
        assert polygon.check_collision(point) == True

        # collision check with point inside at arbitrary time
        point = Point(0.5, 0.5)
        assert polygon.check_collision(point, query_time=5) == True

        # collision check with point outside at arbitrary time
        point = Point(3, 3)
        assert polygon.check_collision(point, query_time=5) == False

        # collision check with line inside
        line = LineString([(0, 0), (1, 1), (2, 2)])
        assert polygon.check_collision(line) == True

        # collision check with line outside
        line = LineString([(4, 4), (3, 3)])
        assert polygon.check_collision(line) == False

        # collision check with line inside at arbitrary time
        line = LineString([(0, 0), (1, 1), (2, 2)])
        assert polygon.check_collision(line, query_time=5) == True

        # collision check with line outside at arbitrary time
        line = LineString([(4, 4), (3, 3)])
        assert polygon.check_collision(line, query_time=5) == False

        # collision check with polygon inside
        other_polygon = ShapelyPolygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        assert polygon.check_collision(other_polygon) == True

        # collision check with polygon outside
        other_polygon = ShapelyPolygon([(4, 4), (4, 3), (3, 3), (3, 4)])
        assert polygon.check_collision(other_polygon) == False

        # collision check with polygon inside at arbitrary time
        other_polygon = ShapelyPolygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        assert polygon.check_collision(other_polygon, query_time=5) == True

        # collision check with polygon touching at arbitrary time
        other_polygon = ShapelyPolygon([(3, 3), (2, 3), (3, 3), (3, 2)])
        assert polygon.check_collision(other_polygon, query_time=5) == True

        # collision check with polygon outside at arbitrary time
        other_polygon = ShapelyPolygon([(3, 3), (4, 3), (3, 3), (3, 5)])
        assert polygon.check_collision(other_polygon, query_time=5) == False

    def test_plot(self):
        fig = plt.figure()

        # Test case 1: No figure provided
        polygon = Polygon(ShapelyPolygon([(0, 0), (1, 1), (1, 0)]))
        polygon.plot()  # Plot the polygon on a new figure

        # Test case 2: Plotting on an existing figure
        polygon = Polygon(ShapelyPolygon([(0, 0), (1, 1), (1, 0)]))
        polygon.plot(fig=fig)  # Plot the polygon on the provided figure

        # Test case 3: Plotting with temporal input arguments
        polygon = Polygon(ShapelyPolygon([(0, 0), (1, 1), (1, 0)]), None, 1.0)
        polygon.plot(
            fig=fig, query_time=5
        )  # Plot the polygon at a specific time on the provided figure

        # Test case 4: Plotting with temporal input arguments
        polygon = Polygon(ShapelyPolygon([(0, 0), (1, 1), (1, 0)]), None, 1.0)
        polygon.plot(fig=fig, query_interval=Interval(0, 10))
