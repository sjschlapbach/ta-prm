import pytest
import json
import os
from shapely.geometry import (
    Point as ShapelyPoint,
    LineString as ShapelyLine,
    Polygon as ShapelyPolygon,
)
from pandas import Interval
from matplotlib import pyplot as plt

from src.obstacles.point import Point


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
        point = Point(geometry=ShapelyPoint(0, 0))
        assert point.geometry == ShapelyPoint(0, 0)
        assert point.time_interval == None
        assert point.radius == 0

        # Test constructor with geometry and time interval
        point = Point(
            geometry=ShapelyPoint(0, 0), time_interval=Interval(0, 10, closed="both")
        )
        assert point.geometry == ShapelyPoint(0, 0)
        assert point.time_interval == Interval(0, 10, closed="both")
        assert point.radius == 0

        # Test constructor with geometry, time interval, and radius
        point = Point(
            geometry=ShapelyPoint(0, 0),
            time_interval=Interval(0, 10, closed="both"),
            radius=1.0,
        )
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
        line = ShapelyLine([(1, 1), (2, 2)])
        assert point.check_collision(line) == False

        # collision check with line passing through
        line = ShapelyLine([(0, 0), (0, 2), (2, 2)])
        assert point.check_collision(line) == True

        # collision check with line on the edge
        line = ShapelyLine([(1, -2), (1, 2)])
        assert point.check_collision(line) == True

        # collision check with line slightly outside the area
        line = ShapelyLine([(1 + 10e-5, -2), (1 + 10e-5, 2)])
        assert point.check_collision(line) == False

    def test_check_collision_with_polygon(self):
        point = self.setup_method()

        # collision check with polygon outside
        polygon = ShapelyPolygon([(1, 1), (1, 2), (2, 2), (2, 1)])
        assert point.check_collision(polygon) == False

        # collision check with polygon containing point
        polygon = ShapelyPolygon([(0, 0), (0, 2), (2, 2), (2, 0)])
        assert point.check_collision(polygon) == True

        # collision check with polygon on the edge
        polygon = ShapelyPolygon([(1, -2), (1, 2), (2, 2)])
        assert point.check_collision(polygon) == True

        # collision check with polygon slightly outside the area
        polygon = ShapelyPolygon([(1 + 10e-5, -2), (1 + 10e-5, 2), (2, 2)])
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
        assert point.is_active(query_time=5) == True
        assert point.check_collision(test_pt, query_time=5) == True
        assert point.is_active(query_interval=in1) == True
        assert point.check_collision(test_pt, query_interval=in1) == True
        assert point.is_active(query_time=15) == False
        assert point.check_collision(test_pt, query_time=15) == False
        assert point.is_active(query_interval=in3) == False
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
        line = ShapelyLine([(0, 0), (1, 1)])
        assert point.check_collision(line, query_time=5) == True
        assert point.check_collision(line, query_interval=in1) == True
        assert point.check_collision(line, query_time=15) == False
        assert point.check_collision(line, query_interval=in3) == False

        # collision check with polygon and different temporal queries
        polygon = ShapelyPolygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        assert point.check_collision(polygon, query_time=5) == True
        assert point.check_collision(polygon, query_interval=in1) == True
        assert point.check_collision(polygon, query_time=15) == False
        assert point.check_collision(polygon, query_interval=in3) == False

    def test_check_collision_without_time_interval(self):
        point = Point(geometry=ShapelyPoint(0, 0), radius=1.0)

        # collision check with point outside
        other_point = ShapelyPoint(1, 1)
        assert point.check_collision(other_point) == False

        # collision check with point inside
        other_point = ShapelyPoint(0, 0)
        assert point.check_collision(other_point) == True

        # collision check with point inside at arbitrary time
        other_point = ShapelyPoint(0, 0)
        assert point.check_collision(other_point, query_time=5) == True

        # collision check with point outside at arbitrary time
        other_point = ShapelyPoint(1, 1)
        assert point.check_collision(other_point, query_time=5) == False

        # collision check with line inside
        line = ShapelyLine([(0, 0), (0, 2), (2, 2)])
        assert point.check_collision(line) == True

        # collision check with line outside
        line = ShapelyLine([(1, 1), (2, 2)])
        assert point.check_collision(line) == False

        # collision check with line inside at arbitrary time
        line = ShapelyLine([(0, 0), (0, 2), (2, 2)])
        assert point.check_collision(line, query_time=5) == True

        # collision check with line outside at arbitrary time
        line = ShapelyLine([(1, 1), (2, 2)])
        assert point.check_collision(line, query_time=5) == False

        # collision check with polygon inside
        polygon = ShapelyPolygon([(0, 0), (0, 2), (2, 2), (2, 0)])
        assert point.check_collision(polygon) == True

        # collision check with polygon outside
        polygon = ShapelyPolygon([(1, 1), (1, 2), (2, 2), (2, 1)])
        assert point.check_collision(polygon) == False

        # collision check with polygon inside at arbitrary time
        polygon = ShapelyPolygon([(0, 0), (0, 2), (2, 2), (2, 0)])
        assert point.check_collision(polygon, query_time=5) == True

        # collision check with polygon outside at arbitrary time
        polygon = ShapelyPolygon([(1, 1), (1, 2), (2, 2), (2, 1)])
        assert point.check_collision(polygon, query_time=5) == False

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

    def test_load_save(self):
        radius = 5.5
        point = ShapelyPoint(1, 2)
        time_interval = Interval(0, 10, closed="left")

        # Test case 1: Convert point object to JSON and back (only geometry)
        pt_1 = Point(geometry=point)
        json_1 = pt_1.export_to_json()
        loaded_1 = Point()
        loaded_1.load_from_json(json_1)
        assert loaded_1.geometry == point
        assert loaded_1.time_interval == None
        assert loaded_1.radius == 0

        # Test case 2: Convert point object to JSON and back (only geometry and time interval)
        pt_2 = Point(geometry=point, time_interval=time_interval)
        json_2 = pt_2.export_to_json()
        loaded_2 = Point()
        loaded_2.load_from_json(json_2)
        assert loaded_2.geometry == point
        assert loaded_2.time_interval == time_interval
        assert loaded_2.radius == 0

        # Test case 3: Convert point object to JSON and back (only geometry, time interval, and radius)
        pt_3 = Point(geometry=point, time_interval=time_interval, radius=radius)
        json_3 = pt_3.export_to_json()
        loaded_3 = Point()
        loaded_3.load_from_json(json_3)
        assert loaded_3.geometry == point
        assert loaded_3.time_interval == time_interval
        assert loaded_3.radius == radius

        # Test case 4: Convert point object to JSON and back (only time interval and radius)
        pt_4 = Point(time_interval=time_interval, radius=radius)
        json_4 = pt_4.export_to_json()
        loaded_4 = Point()
        loaded_4.load_from_json(json_4)
        assert loaded_4.geometry == None
        assert loaded_4.time_interval == time_interval
        assert loaded_4.radius == radius

        # Test case 5: Convert point object to JSON and back (only geometry and radius)
        pt_5 = Point(geometry=point, radius=radius)
        json_5 = pt_5.export_to_json()
        loaded_5 = Point()
        loaded_5.load_from_json(json_5)
        assert loaded_5.geometry == point
        assert loaded_5.time_interval == None
        assert loaded_5.radius == radius

        # Test case 6: Convert point object to JSON and back (only radius)
        pt_6 = Point(radius=radius)
        json_6 = pt_6.export_to_json()
        loaded_6 = Point()
        loaded_6.load_from_json(json_6)
        assert loaded_6.geometry == None
        assert loaded_6.time_interval == None
        assert loaded_6.radius == radius

        # Test case 7: Convert point object to JSON and back (only geometry and time interval)
        pt_7 = Point(geometry=point, time_interval=time_interval)
        json_7 = pt_7.export_to_json()
        loaded_7 = Point()
        loaded_7.load_from_json(json_7)
        assert loaded_7.geometry == point
        assert loaded_7.time_interval == time_interval
        assert loaded_7.radius == 0

        # Test case 8: Convert point object to JSON, save and load from file
        pt_8 = Point(geometry=point, time_interval=time_interval, radius=radius)
        json_8 = pt_8.export_to_json()

        with open("test_point_saving.txt", "w") as f:
            json.dump(json_8, f)

        with open("test_point_saving.txt", "r") as f:
            json_obj8_loaded = json.load(f)

        loaded_8 = Point()
        loaded_8.load_from_json(json_obj8_loaded)
        assert loaded_8.geometry == point
        assert loaded_8.time_interval == time_interval
        assert loaded_8.radius == radius

        os.remove("test_point_saving.txt")
