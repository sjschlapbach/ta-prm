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
from src.util.recurrence import Recurrence


class TestPoint:
    def setup_method(self):
        point = Point(
            geometry=ShapelyPoint(0, 0),
            time_interval=Interval(0, 10, closed="both"),
            radius=1.0,
        )
        return point

    def test_setup(self):
        # Test constructor with out anything
        point = Point()
        assert point.geometry == None
        assert point.time_interval == None
        assert point.radius == 0
        assert point.recurrence == Recurrence.NONE

        # Test constructor with geometry
        point = Point(geometry=ShapelyPoint(0, 0))
        assert point.geometry == ShapelyPoint(0, 0)
        assert point.time_interval == None
        assert point.radius == 0
        assert point.recurrence == Recurrence.NONE

        # Test constructor with geometry and time interval
        point = Point(
            geometry=ShapelyPoint(0, 0), time_interval=Interval(0, 10, closed="both")
        )
        assert point.geometry == ShapelyPoint(0, 0)
        assert point.time_interval == Interval(0, 10, closed="both")
        assert point.radius == 0
        assert point.recurrence == Recurrence.NONE

        # Test constructor with geometry, time interval, and radius
        point = Point(
            geometry=ShapelyPoint(0, 0),
            time_interval=Interval(0, 10, closed="both"),
            radius=1.0,
        )
        assert point.geometry == ShapelyPoint(0, 0)
        assert point.time_interval == Interval(0, 10, closed="both")
        assert point.radius == 1.0
        assert point.recurrence == Recurrence.NONE

        # Test constructor with geometry, time interval, radius, and recurrence
        point = Point(
            geometry=ShapelyPoint(0, 0),
            time_interval=Interval(0, 10, closed="both"),
            radius=1.0,
            recurrence=Recurrence.MINUTELY,
        )
        assert point.geometry == ShapelyPoint(0, 0)
        assert point.time_interval == Interval(0, 10, closed="both")
        assert point.radius == 1.0
        assert point.recurrence == Recurrence.MINUTELY

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

        # check collision with recurring point
        point_rec = Point(
            geometry=ShapelyPoint(0, 0),
            time_interval=Interval(5, 15, closed="both"),
            radius=1.0,
            recurrence=Recurrence.MINUTELY,
        )
        colliding_point = ShapelyPoint(0, 0)
        non_colliding_point = ShapelyPoint(2, 2)
        assert point_rec.check_collision(colliding_point, query_time=0) == False
        assert point_rec.check_collision(non_colliding_point, query_time=0) == False
        assert point_rec.check_collision(colliding_point, query_time=10) == True
        assert point_rec.check_collision(non_colliding_point, query_time=10) == False
        assert point_rec.check_collision(colliding_point, query_time=20) == False
        assert point_rec.check_collision(non_colliding_point, query_time=20) == False

        assert (
            point_rec.check_collision(colliding_point, query_interval=Interval(0, 3))
            == False
        )
        assert (
            point_rec.check_collision(
                non_colliding_point, query_interval=Interval(0, 3)
            )
            == False
        )
        assert (
            point_rec.check_collision(colliding_point, query_interval=Interval(3, 10))
            == True
        )
        assert (
            point_rec.check_collision(
                non_colliding_point, query_interval=Interval(3, 10)
            )
            == False
        )
        assert (
            point_rec.check_collision(colliding_point, query_interval=Interval(10, 15))
            == True
        )
        assert (
            point_rec.check_collision(
                non_colliding_point, query_interval=Interval(10, 15)
            )
            == False
        )

        assert point_rec.check_collision(colliding_point, query_time=120) == False
        assert point_rec.check_collision(non_colliding_point, query_time=120) == False
        assert point_rec.check_collision(colliding_point, query_time=130) == True
        assert point_rec.check_collision(non_colliding_point, query_time=130) == False
        assert point_rec.check_collision(colliding_point, query_time=140) == False
        assert point_rec.check_collision(non_colliding_point, query_time=140) == False

        assert (
            point_rec.check_collision(
                colliding_point, query_interval=Interval(120, 123)
            )
            == False
        )
        assert (
            point_rec.check_collision(
                non_colliding_point, query_interval=Interval(120, 123)
            )
            == False
        )
        assert (
            point_rec.check_collision(
                colliding_point, query_interval=Interval(125, 130)
            )
            == True
        )
        assert (
            point_rec.check_collision(
                non_colliding_point, query_interval=Interval(125, 130)
            )
            == False
        )
        assert (
            point_rec.check_collision(
                colliding_point, query_interval=Interval(130, 140)
            )
            == True
        )
        assert (
            point_rec.check_collision(
                non_colliding_point, query_interval=Interval(130, 140)
            )
            == False
        )

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
        with pytest.raises(TypeError):
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
        loaded_1 = Point(json_data=json_1)
        assert loaded_1.geometry == point
        assert loaded_1.time_interval == None
        assert loaded_1.radius == 0

        # Test case 2: Convert point object to JSON and back (only geometry and time interval)
        pt_2 = Point(geometry=point, time_interval=time_interval)
        json_2 = pt_2.export_to_json()
        loaded_2 = Point(json_data=json_2)
        assert loaded_2.geometry == point
        assert loaded_2.time_interval == time_interval
        assert loaded_2.radius == 0

        # Test case 3: Convert point object to JSON and back (only geometry, time interval, and radius)
        pt_3 = Point(geometry=point, time_interval=time_interval, radius=radius)
        json_3 = pt_3.export_to_json()
        loaded_3 = Point(json_data=json_3)
        assert loaded_3.geometry == point
        assert loaded_3.time_interval == time_interval
        assert loaded_3.radius == radius

        # Test case 4: Convert point object to JSON and back (only time interval and radius)
        pt_4 = Point(time_interval=time_interval, radius=radius)
        json_4 = pt_4.export_to_json()
        loaded_4 = Point(json_data=json_4)
        assert loaded_4.geometry == None
        assert loaded_4.time_interval == time_interval
        assert loaded_4.radius == radius

        # Test case 5: Convert point object to JSON and back (only geometry and radius)
        pt_5 = Point(geometry=point, radius=radius)
        json_5 = pt_5.export_to_json()
        loaded_5 = Point(json_data=json_5)
        assert loaded_5.geometry == point
        assert loaded_5.time_interval == None
        assert loaded_5.radius == radius

        # Test case 6: Convert point object to JSON and back (only radius)
        pt_6 = Point(radius=radius)
        json_6 = pt_6.export_to_json()
        loaded_6 = Point(json_data=json_6)
        assert loaded_6.geometry == None
        assert loaded_6.time_interval == None
        assert loaded_6.radius == radius

        # Test case 7: Convert point object to JSON and back (only geometry and time interval)
        pt_7 = Point(geometry=point, time_interval=time_interval)
        json_7 = pt_7.export_to_json()
        loaded_7 = Point(json_data=json_7)
        assert loaded_7.geometry == point
        assert loaded_7.time_interval == time_interval
        assert loaded_7.radius == 0

        # Test case 8: Convert point object to JSON and back (geometry, time interval, radius, and recurrence)
        pt_8 = Point(
            geometry=point,
            time_interval=time_interval,
            radius=radius,
            recurrence=Recurrence.MINUTELY,
        )
        json_8 = pt_8.export_to_json()
        loaded_8 = Point(json_data=json_8)
        assert loaded_8.geometry == point
        assert loaded_8.time_interval == time_interval
        assert loaded_8.radius == radius
        assert loaded_8.recurrence == Recurrence.MINUTELY

        # Test case 9: Convert point object to JSON, save and load from file
        pt_8 = Point(
            geometry=point,
            time_interval=time_interval,
            radius=radius,
            recurrence=Recurrence.HOURLY,
        )
        json_8 = pt_8.export_to_json()

        with open("test_point_saving.txt", "w") as f:
            json.dump(json_8, f)

        with open("test_point_saving.txt", "r") as f:
            json_obj8_loaded = json.load(f)

        loaded_8 = Point(json_data=json_obj8_loaded)
        assert loaded_8.geometry == point
        assert loaded_8.time_interval == time_interval
        assert loaded_8.radius == radius
        assert loaded_8.recurrence == Recurrence.HOURLY

        os.remove("test_point_saving.txt")

    def test_copy(self):
        point = Point(
            geometry=ShapelyPoint(0, 0),
            time_interval=Interval(0, 10, closed="both"),
            radius=1.0,
            recurrence=Recurrence.MINUTELY,
        )
        copy = point.copy()

        assert copy.geometry == point.geometry
        assert copy.time_interval == point.time_interval
        assert copy.radius == point.radius
        assert copy.recurrence == point.recurrence
        assert copy != point

    def test_random_generation(self):
        ## Test random point generation with different inputs.
        ## For each combination of inputs, multiple points are generated and tested for correct parameters.

        # Function default values
        min_interval_default = 0
        max_interval_default = 100

        # Test case 1 - default function
        min_x = 0
        max_x = 100
        min_y = 0
        max_y = 100
        min_radius = 0.1
        max_radius = 10

        pt1 = Point.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
        )
        assert pt1.geometry.x >= min_x and pt1.geometry.x <= max_x
        assert pt1.geometry.y >= min_y and pt1.geometry.y <= max_y
        assert pt1.time_interval == None or (
            pt1.time_interval.left >= min_interval_default
            and pt1.time_interval.right <= max_interval_default
        )
        assert pt1.radius >= min_radius and pt1.radius <= max_radius
        assert pt1.recurrence == Recurrence.NONE

        min_x = -100
        max_x = 100
        min_y = -100
        max_y = 100
        min_radius = 10
        max_radius = 20
        pt2 = Point.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
        )
        assert pt2.geometry.x >= min_x and pt2.geometry.x <= max_x
        assert pt2.geometry.y >= min_y and pt2.geometry.y <= max_y
        assert pt2.time_interval == None or (
            pt2.time_interval.left >= min_interval_default
            and pt2.time_interval.right <= max_interval_default
        )
        assert pt2.radius >= min_radius and pt2.radius <= max_radius
        assert pt2.recurrence == Recurrence.NONE

        # Test case 2 - only static points without recurrence
        pt3 = Point.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            only_static=True,
        )
        assert pt3.geometry.x >= min_x and pt3.geometry.x <= max_x
        assert pt3.geometry.y >= min_y and pt3.geometry.y <= max_y
        assert pt3.time_interval == None
        assert pt3.radius >= min_radius and pt3.radius <= max_radius
        assert pt3.recurrence == Recurrence.NONE

        min_x = -200
        max_x = 0
        min_y = -100
        max_y = 100
        min_radius = 20
        max_radius = 30
        pt4 = Point.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            only_static=True,
        )
        assert pt4.geometry.x >= min_x and pt4.geometry.x <= max_x
        assert pt4.geometry.y >= min_y and pt4.geometry.y <= max_y
        assert pt4.time_interval == None
        assert pt4.radius >= min_radius and pt4.radius <= max_radius
        assert pt4.recurrence == Recurrence.NONE

        # Test case 3 - random dynamic points with recurrence
        min_x = 0
        max_x = 100
        min_y = 0
        max_y = 100
        min_radius = 0.1
        max_radius = 10

        pt5 = Point.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            only_dynamic=True,
            random_recurrence=True,
        )
        assert pt5.geometry.x >= min_x and pt5.geometry.x <= max_x
        assert pt5.geometry.y >= min_y and pt5.geometry.y <= max_y
        assert pt5.time_interval.left >= min_interval_default
        assert pt5.time_interval.right <= max_interval_default
        assert pt5.radius >= min_radius and pt5.radius <= max_radius
        assert pt5.recurrence in [
            Recurrence.NONE,
            Recurrence.MINUTELY,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]

        pt6 = Point.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            only_dynamic=True,
            random_recurrence=True,
        )
        assert pt6.geometry.x >= min_x and pt6.geometry.x <= max_x
        assert pt6.geometry.y >= min_y and pt6.geometry.y <= max_y
        assert pt6.time_interval.left >= min_interval_default
        assert pt6.time_interval.right <= max_interval_default
        assert pt6.radius >= min_radius and pt6.radius <= max_radius
        assert pt6.recurrence in [
            Recurrence.NONE,
            Recurrence.MINUTELY,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]

        pt7 = Point.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            only_dynamic=True,
            random_recurrence=True,
        )
        assert pt7.geometry.x >= min_x and pt7.geometry.x <= max_x
        assert pt7.geometry.y >= min_y and pt7.geometry.y <= max_y
        assert pt7.time_interval.left >= min_interval_default
        assert pt7.time_interval.right <= max_interval_default
        assert pt7.radius >= min_radius and pt7.radius <= max_radius
        assert pt7.recurrence in [
            Recurrence.NONE,
            Recurrence.MINUTELY,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]

        # Test case 4 - dynamic points with custom time interval
        min_interval = 100
        max_interval = 200

        pt8 = Point.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            min_interval=min_interval,
            max_interval=max_interval,
            only_dynamic=True,
            random_recurrence=True,
        )
        assert pt8.geometry.x >= min_x and pt8.geometry.x <= max_x
        assert pt8.geometry.y >= min_y and pt8.geometry.y <= max_y
        assert pt8.time_interval.left >= min_interval
        assert pt8.time_interval.right <= max_interval
        assert pt8.radius >= min_radius and pt8.radius <= max_radius
        assert pt8.recurrence in [
            Recurrence.NONE,
            Recurrence.MINUTELY,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]

        pt9 = Point.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            min_interval=min_interval,
            max_interval=max_interval,
            only_dynamic=True,
            random_recurrence=True,
        )
        assert pt9.geometry.x >= min_x and pt9.geometry.x <= max_x
        assert pt9.geometry.y >= min_y and pt9.geometry.y <= max_y
        assert pt9.time_interval.left >= min_interval
        assert pt9.time_interval.right <= max_interval
        assert pt9.radius >= min_radius and pt9.radius <= max_radius
        assert pt9.recurrence in [
            Recurrence.NONE,
            Recurrence.MINUTELY,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]

        pt10 = Point.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            min_interval=min_interval,
            max_interval=max_interval,
            only_dynamic=True,
        )
        assert pt10.geometry.x >= min_x and pt10.geometry.x <= max_x
        assert pt10.geometry.y >= min_y and pt10.geometry.y <= max_y
        assert pt10.time_interval.left >= min_interval
        assert pt10.time_interval.right <= max_interval
        assert pt10.radius >= min_radius and pt10.radius <= max_radius
        assert pt10.recurrence == Recurrence.NONE

        # Test case 5 - dynamic points with custom time interval, radius and random recurrence
        min_x = 1000
        max_x = 2000
        min_y = 3000
        max_y = 4000
        min_radius = 3
        max_radius = 40
        min_interval = 100
        max_interval = 200

        pt11 = Point.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            min_interval=min_interval,
            max_interval=max_interval,
            random_recurrence=True,
        )

        assert pt11.geometry.x >= min_x and pt11.geometry.x <= max_x
        assert pt11.geometry.y >= min_y and pt11.geometry.y <= max_y
        assert pt11.radius >= min_radius and pt11.radius <= max_radius

        # check if point is static or not and test accordingly
        if pt11.time_interval == None:
            assert pt11.recurrence == Recurrence.NONE
        else:
            assert pt11.time_interval.left >= min_interval
            assert pt11.time_interval.right <= max_interval
            assert pt11.recurrence in [
                Recurrence.NONE,
                Recurrence.MINUTELY,
                Recurrence.HOURLY,
                Recurrence.DAILY,
            ]

        pt12 = Point.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            min_interval=min_interval,
            max_interval=max_interval,
            random_recurrence=True,
        )

        assert pt12.geometry.x >= min_x and pt12.geometry.x <= max_x
        assert pt12.geometry.y >= min_y and pt12.geometry.y <= max_y
        assert pt12.radius >= min_radius and pt12.radius <= max_radius

        # check if point is static or not and test accordingly
        if pt12.time_interval == None:
            assert pt12.recurrence == Recurrence.NONE
        else:
            assert pt12.time_interval.left >= min_interval
            assert pt12.time_interval.right <= max_interval
            assert pt12.recurrence in [
                Recurrence.NONE,
                Recurrence.MINUTELY,
                Recurrence.HOURLY,
                Recurrence.DAILY,
            ]

        pt13 = Point.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            min_interval=min_interval,
            max_interval=max_interval,
            random_recurrence=True,
        )

        assert pt13.geometry.x >= min_x and pt13.geometry.x <= max_x
        assert pt13.geometry.y >= min_y and pt13.geometry.y <= max_y
        assert pt13.radius >= min_radius and pt13.radius <= max_radius

        # check if point is static or not and test accordingly
        if pt13.time_interval == None:
            assert pt13.recurrence == Recurrence.NONE
        else:
            assert pt13.time_interval.left >= min_interval
            assert pt13.time_interval.right <= max_interval
            assert pt13.recurrence in [
                Recurrence.NONE,
                Recurrence.MINUTELY,
                Recurrence.HOURLY,
                Recurrence.DAILY,
            ]
