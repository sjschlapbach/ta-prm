import pytest
import json
import os
from shapely.geometry import Point, LineString as ShapelyLine, Polygon as ShapelyPolygon
from pandas import Interval
from matplotlib import pyplot as plt

from src.obstacles.line import Line
from src.util.recurrence import Recurrence


class TestLine:
    def setup_method(self):
        line = Line(
            geometry=ShapelyLine([(0, 0), (1, 1)]),
            time_interval=Interval(0, 10, closed="both"),
            radius=1.0,
        )
        return line

    def test_setup(self):
        # Test constructor with out anything
        line = Line()
        assert line.geometry == None
        assert line.time_interval == None
        assert line.radius == 0
        assert line.recurrence == Recurrence.NONE

        # Test constructor with start and end points
        line = Line(geometry=ShapelyLine([(0, 0), (1, 1)]))
        assert line.geometry == ShapelyLine([(0, 0), (1, 1)])
        assert line.time_interval == None
        assert line.radius == 0
        assert line.recurrence == Recurrence.NONE

        # Test constructor with start and end points, time interval, and radius
        line = Line(
            geometry=ShapelyLine([(0, 0), (1, 1)]),
            time_interval=Interval(0, 10, closed="both"),
            radius=1.0,
        )
        assert line.geometry == ShapelyLine([(0, 0), (1, 1)])
        assert line.time_interval == Interval(0, 10, closed="both")
        assert line.radius == 1.0
        assert line.recurrence == Recurrence.NONE

        # Test constructor with start and end points, time interval, radius, and recurrence
        line = Line(
            geometry=ShapelyLine([(0, 0), (1, 1)]),
            time_interval=Interval(0, 10, closed="both"),
            radius=1.0,
            recurrence=Recurrence.MINUTELY,
        )
        assert line.geometry == ShapelyLine([(0, 0), (1, 1)])
        assert line.time_interval == Interval(0, 10, closed="both")
        assert line.radius == 1.0
        assert line.recurrence == Recurrence.MINUTELY

    def test_set_geometry(self):
        line = self.setup_method()
        line.set_geometry([(1, 2), (1, 1)])
        assert line.geometry == ShapelyLine([(1, 2), (1, 1)])

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
        other_line = ShapelyLine([(2, 2), (3, 3)])
        assert line.check_collision(other_line) == False

        # collision check with line passing through
        other_line = ShapelyLine([(0, 0), (1, 1), (2, 2)])
        assert line.check_collision(other_line) == True

        # collision check with line on the edge
        other_line = ShapelyLine([(0.5, 0.5), (1.5, 1.5)])
        assert line.check_collision(other_line) == True

        # collision check with line inside radius region
        other_line = ShapelyLine([(1.5, 1.5), (2.5, 2.5)])
        assert line.check_collision(other_line) == True

        # collision check with line slightly outside the line
        other_line = ShapelyLine([(1.75, 1.75), (2.5, 2.5)])
        assert line.check_collision(other_line) == False

        # check collision with recurring line
        line_rec = Line(
            geometry=ShapelyLine([(0, 0), (1, 1)]),
            time_interval=Interval(5, 15, closed="both"),
            recurrence=Recurrence.MINUTELY,
        )
        colliding_line = ShapelyLine([(0.5, 0.5), (1.5, 1.5)])
        non_colliding_line = ShapelyLine([(2, 2), (3, 3)])

        assert line_rec.check_collision(colliding_line, query_time=0) == False
        assert line_rec.check_collision(non_colliding_line, query_time=0) == False
        assert line_rec.check_collision(colliding_line, query_time=10) == True
        assert line_rec.check_collision(non_colliding_line, query_time=10) == False
        assert line_rec.check_collision(colliding_line, query_time=20) == False
        assert line_rec.check_collision(non_colliding_line, query_time=20) == False

        assert (
            line_rec.check_collision(colliding_line, query_interval=Interval(0, 3))
            == False
        )
        assert (
            line_rec.check_collision(non_colliding_line, query_interval=Interval(0, 3))
            == False
        )
        assert (
            line_rec.check_collision(colliding_line, query_interval=Interval(3, 10))
            == True
        )
        assert (
            line_rec.check_collision(non_colliding_line, query_interval=Interval(3, 10))
            == False
        )
        assert (
            line_rec.check_collision(colliding_line, query_interval=Interval(10, 15))
            == True
        )
        assert (
            line_rec.check_collision(
                non_colliding_line, query_interval=Interval(10, 15)
            )
            == False
        )

        assert line_rec.check_collision(colliding_line, query_time=120) == False
        assert line_rec.check_collision(non_colliding_line, query_time=120) == False
        assert line_rec.check_collision(colliding_line, query_time=130) == True
        assert line_rec.check_collision(non_colliding_line, query_time=130) == False
        assert line_rec.check_collision(colliding_line, query_time=140) == False
        assert line_rec.check_collision(non_colliding_line, query_time=140) == False

        assert (
            line_rec.check_collision(colliding_line, query_interval=Interval(120, 123))
            == False
        )
        assert (
            line_rec.check_collision(
                non_colliding_line, query_interval=Interval(120, 123)
            )
            == False
        )
        assert (
            line_rec.check_collision(colliding_line, query_interval=Interval(125, 130))
            == True
        )
        assert (
            line_rec.check_collision(
                non_colliding_line, query_interval=Interval(125, 130)
            )
            == False
        )
        assert (
            line_rec.check_collision(colliding_line, query_interval=Interval(130, 140))
            == True
        )
        assert (
            line_rec.check_collision(
                non_colliding_line, query_interval=Interval(130, 140)
            )
            == False
        )

    def test_check_collision_with_polygon(self):
        line = self.setup_method()

        # collision check with polygon outside
        polygon = ShapelyPolygon([(2, 2), (2, 3), (3, 3), (3, 2)])
        assert line.check_collision(polygon) == False

        # collision check with polygon containing line
        polygon = ShapelyPolygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        assert line.check_collision(polygon) == True

        # collision check with polygon on the edge
        polygon = ShapelyPolygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)])
        assert line.check_collision(polygon) == True

        # collision check with polygon inside radius region
        polygon = ShapelyPolygon([(1.5, 1.5), (1.5, 2.5), (2.5, 2.5), (2.5, 1.5)])
        assert line.check_collision(polygon) == True

        # collision check with polygon slightly outside the line
        polygon = ShapelyPolygon([(1.75, 1.75), (1.75, 2.5), (2.5, 2.5), (2.5, 1.75)])
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
        test_line = ShapelyLine([(0.5, 0.5), (1.5, 1.5)])

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
        test_line = ShapelyLine([(5.5, 5.5), (6.5, 6.5)])
        assert line.check_collision(test_line, query_time=5) == False
        assert line.check_collision(test_line, query_interval=in1) == False
        assert line.check_collision(test_line, query_time=15) == False
        assert line.check_collision(test_line, query_interval=in3) == False

        # collision check with line on the edge of spatial area
        test_line = ShapelyLine([(1, 1), (2, 2)])
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
        polygon = ShapelyPolygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        assert line.check_collision(polygon, query_time=5) == True
        assert line.check_collision(polygon, query_interval=in1) == True
        assert line.check_collision(polygon, query_time=15) == False
        assert line.check_collision(polygon, query_interval=in3) == False

    def test_check_collision_without_time_interval(self):
        line = Line(geometry=ShapelyLine([(0, 0), (1, 1)]), radius=1.0)

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
        other_line = ShapelyLine([(0, 0), (1, 1), (2, 2)])
        assert line.check_collision(other_line) == True

        # collision check with line outside
        other_line = ShapelyLine([(2, 2), (3, 3)])
        assert line.check_collision(other_line) == False

        # collision check with line inside at arbitrary time
        other_line = ShapelyLine([(0, 0), (1, 1), (2, 2)])
        assert line.check_collision(other_line, query_time=5) == True

        # collision check with line outside at arbitrary time
        other_line = ShapelyLine([(2, 2), (3, 3)])
        assert line.check_collision(other_line, query_time=5) == False

        # collision check with polygon inside
        polygon = ShapelyPolygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        assert line.check_collision(polygon) == True

        # collision check with polygon outside
        polygon = ShapelyPolygon([(2, 2), (2, 3), (3, 3), (3, 2)])
        assert line.check_collision(polygon) == False

        # collision check with polygon inside at arbitrary time
        polygon = ShapelyPolygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        assert line.check_collision(polygon, query_time=5) == True

        # collision check with polygon outside at arbitrary time
        polygon = ShapelyPolygon([(2, 2), (2, 3), (3, 3), (3, 2)])
        assert line.check_collision(polygon, query_time=5) == False

    def test_plot(self):
        fig = plt.figure()

        # Test case 1: No query time or query interval provided
        line = Line(
            geometry=ShapelyLine([(0, 0), (1, 1)]),
            time_interval=Interval(0, 10, closed="both"),
            radius=1.0,
        )
        line.plot(fig=fig)  # Plot the line

        # Test case 2: Query time is provided, but line has no time interval
        line = Line(geometry=ShapelyLine([(2, 2), (3, 3)]))
        line.plot(query_time=5, fig=fig)

        # Test case 3: Query time is within the line's time interval
        line = Line(
            geometry=ShapelyLine([(0, 0), (1, 1)]),
            time_interval=Interval(0, 10, closed="both"),
            radius=1.0,
        )
        line.plot(query_time=5, fig=fig)  # Plot the line at query time 5

        # Test case 4: Query interval overlaps with the line's time interval
        line = Line(
            geometry=ShapelyLine([(0, 0), (1, 1)]),
            time_interval=Interval(0, 10, closed="both"),
            radius=1.0,
        )
        line.plot(query_interval=Interval(5, 15), fig=fig)

        # Test case 5: Query time is outside the line's time interval
        line = Line(
            geometry=ShapelyLine([(0, 0), (1, 1)]),
            time_interval=Interval(0, 10, closed="both"),
            radius=1.0,
        )
        line.plot(query_time=15, fig=fig)

        # Test case 6: Query interval does not overlap with the line's time interval
        line = Line(
            geometry=ShapelyLine([(0, 0), (1, 1)]),
            time_interval=Interval(0, 10, closed="both"),
            radius=1.0,
        )
        line.plot(query_interval=Interval(15, 25), fig=fig)

        # Test case 7: No figure provided
        line = Line(
            geometry=ShapelyLine([(0, 0), (1, 1)]),
            time_interval=Interval(0, 10, closed="both"),
            radius=1.0,
        )
        line.plot()  # Plot the line on a new figure

    def test_load_save(self):
        radius = 5.5
        line = ShapelyLine([(0, 0), (1, 1)])
        time_interval = Interval(0, 10, closed="left")

        # Test case 1: Convert line object to JSON and back (only geometry)
        ln_1 = Line(geometry=line)
        json_1 = ln_1.export_to_json()
        loaded_1 = Line(json_data=json_1)
        assert loaded_1.geometry == line
        assert loaded_1.time_interval == None
        assert loaded_1.radius == 0

        # Test case 2: Convert line object to JSON and back (only geometry and time interval)
        ln_2 = Line(geometry=line, time_interval=time_interval)
        json_2 = ln_2.export_to_json()
        loaded_2 = Line(json_data=json_2)
        assert loaded_2.geometry == line
        assert loaded_2.time_interval == time_interval
        assert loaded_2.radius == 0

        # Test case 3: Convert line object to JSON and back (only geometry, time interval, and radius)
        ln_3 = Line(geometry=line, time_interval=time_interval, radius=radius)
        json_3 = ln_3.export_to_json()
        loaded_3 = Line(json_data=json_3)
        assert loaded_3.geometry == line
        assert loaded_3.time_interval == time_interval
        assert loaded_3.radius == radius

        # Test case 4: Convert line object to JSON and back (only time interval and radius)
        ln_4 = Line(time_interval=time_interval, radius=radius)
        json_4 = ln_4.export_to_json()
        loaded_4 = Line(json_data=json_4)
        assert loaded_4.geometry == None
        assert loaded_4.time_interval == time_interval
        assert loaded_4.radius == radius

        # Test case 5: Convert line object to JSON and back (only geometry and radius)
        ln_5 = Line(geometry=line, radius=radius)
        json_5 = ln_5.export_to_json()
        loaded_5 = Line(json_data=json_5)
        assert loaded_5.geometry == line
        assert loaded_5.time_interval == None
        assert loaded_5.radius == radius

        # Test case 6: Convert line object to JSON and back (only radius)
        ln_6 = Line(radius=radius)
        json_6 = ln_6.export_to_json()
        loaded_6 = Line(json_data=json_6)
        assert loaded_6.geometry == None
        assert loaded_6.time_interval == None
        assert loaded_6.radius == radius

        # Test case 7: Convert line object to JSON and back (only geometry and time interval)
        ln_7 = Line(geometry=line, time_interval=time_interval)
        json_7 = ln_7.export_to_json()
        loaded_7 = Line(json_data=json_7)
        assert loaded_7.geometry == line
        assert loaded_7.time_interval == time_interval
        assert loaded_7.radius == 0

        # Tets case 8: Convert line object to JSON and back (geometry, time interval, radius, and recurrence)
        ln_8 = Line(
            geometry=line,
            time_interval=time_interval,
            radius=radius,
            recurrence=Recurrence.MINUTELY,
        )
        json_8 = ln_8.export_to_json()
        loaded_8 = Line(json_data=json_8)
        assert loaded_8.geometry == line
        assert loaded_8.time_interval == time_interval
        assert loaded_8.radius == radius
        assert loaded_8.recurrence == Recurrence.MINUTELY

        # Test case 9: Convert line object to JSON, save and load from file
        ln_8 = Line(
            geometry=line,
            time_interval=time_interval,
            radius=radius,
            recurrence=Recurrence.DAILY,
        )
        json_8 = ln_8.export_to_json()

        with open("test_line_saving.txt", "w") as f:
            json.dump(json_8, f)

        with open("test_line_saving.txt", "r") as f:
            json_obj8_loaded = json.load(f)

        loaded_8 = Line(json_data=json_obj8_loaded)
        assert loaded_8.geometry == line
        assert loaded_8.time_interval == time_interval
        assert loaded_8.radius == radius
        assert loaded_8.recurrence == Recurrence.DAILY

        os.remove("test_line_saving.txt")
