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

from src.obstacles.polygon import Polygon
from src.util.recurrence import Recurrence


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
        assert polygon.recurrence == Recurrence.NONE

        # Test constructor with points
        polygon = Polygon(geometry=ShapelyPolygon([(0, 0), (1, 1), (1, 0)]))
        assert polygon.geometry == ShapelyPolygon([(0, 0), (1, 1), (1, 0)])

        # Test constructor with start and end points
        poly = Polygon(
            geometry=ShapelyPolygon([(0, 0), (1, 1), (1, 0)]),
            time_interval=Interval(0, 10),
        )
        assert poly.geometry == ShapelyPolygon([(0, 0), (1, 1), (1, 0)])
        assert poly.time_interval == Interval(0, 10)
        assert poly.radius == 0
        assert poly.recurrence == Recurrence.NONE

        # Test constructor with start and end points, time interval, and radius
        poly = Polygon(
            geometry=ShapelyPolygon([(0, 0), (1, 1), (3, 3)]),
            time_interval=Interval(0, 10, closed="both"),
            radius=1.0,
        )
        assert poly.geometry == ShapelyPolygon([(0, 0), (1, 1), (3, 3)])
        assert poly.time_interval == Interval(0, 10, closed="both")
        assert poly.radius == 1.0
        assert poly.recurrence == Recurrence.NONE

        # Test constructor with start and end points, time interval, radius, and recurrence
        poly = Polygon(
            geometry=ShapelyPolygon([(0, 0), (1, 1), (3, 3)]),
            time_interval=Interval(0, 10, closed="both"),
            radius=1.0,
            recurrence=Recurrence.MINUTELY,
        )
        assert poly.geometry == ShapelyPolygon([(0, 0), (1, 1), (3, 3)])
        assert poly.time_interval == Interval(0, 10, closed="both")
        assert poly.radius == 1.0
        assert poly.recurrence == Recurrence.MINUTELY

    def test_set_geometry(self):
        polygon = self.setup_method()
        polygon.set_geometry([(1, 2), (2, 2), (2, 1)])
        assert polygon.geometry == ShapelyPolygon([(1, 2), (2, 2), (2, 1)])

    def test_check_collision_with_point(self):
        polygon = self.setup_method()

        # collision check with point outside
        point = ShapelyPoint(2, 2)
        assert polygon.check_collision(point) == False

        # collision check with point inside
        point = ShapelyPoint(0.5, 0.5)
        assert polygon.check_collision(point) == True

        # collision check with point inside
        point = ShapelyPoint(0.75, 0.5)
        assert polygon.check_collision(point) == True

        # collision check with point on the edge
        point = ShapelyPoint(1, 1)
        assert polygon.check_collision(point) == True

        # collision check with point inside radius region
        point = ShapelyPoint(1.5, 1.5)
        assert polygon.check_collision(point) == True

        # collision check with point slightly outside the polygon
        point = ShapelyPoint(1.75, 1.75)
        assert polygon.check_collision(point) == False

    def test_check_collision_with_line_string(self):
        polygon = self.setup_method()

        # collision check with line outside
        line = ShapelyLine([(2, 2), (3, 3)])
        assert polygon.check_collision(line) == False

        # collision check with line passing through
        line = ShapelyLine([(0, 0), (1, 1), (2, 2)])
        assert polygon.check_collision(line) == True

        # collision check with line on the edge
        line = ShapelyLine([(0.5, 0.5), (1.5, 1.5)])
        assert polygon.check_collision(line) == True

        # collision check with line inside radius region
        line = ShapelyLine([(1.5, 1.5), (2.5, 2.5)])
        assert polygon.check_collision(line) == True

        # collision check with line slightly outside the polygon
        line = ShapelyLine([(1.75, 1.75), (2.5, 2.5)])
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

        # check collision with recurring polygon
        poly_rec = Polygon(
            geometry=ShapelyPolygon([(0, 0), (1, 1), (1, 0)]),
            time_interval=Interval(5, 15, closed="both"),
            radius=1.0,
            recurrence=Recurrence.MINUTELY,
        )
        colliding_poly = ShapelyPolygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        non_colliding_poly = ShapelyPolygon([(3, 3), (3, 4), (4, 4), (4, 3)])

        assert poly_rec.check_collision(colliding_poly, query_time=0) == False
        assert poly_rec.check_collision(non_colliding_poly, query_time=0) == False
        assert poly_rec.check_collision(colliding_poly, query_time=10) == True
        assert poly_rec.check_collision(non_colliding_poly, query_time=10) == False
        assert poly_rec.check_collision(colliding_poly, query_time=20) == False
        assert poly_rec.check_collision(non_colliding_poly, query_time=20) == False

        assert (
            poly_rec.check_collision(colliding_poly, query_interval=Interval(0, 3))
            == False
        )
        assert (
            poly_rec.check_collision(non_colliding_poly, query_interval=Interval(0, 3))
            == False
        )
        assert (
            poly_rec.check_collision(colliding_poly, query_interval=Interval(3, 10))
            == True
        )
        assert (
            poly_rec.check_collision(non_colliding_poly, query_interval=Interval(3, 10))
            == False
        )
        assert (
            poly_rec.check_collision(colliding_poly, query_interval=Interval(10, 15))
            == True
        )
        assert (
            poly_rec.check_collision(
                non_colliding_poly, query_interval=Interval(10, 15)
            )
            == False
        )

        assert poly_rec.check_collision(colliding_poly, query_time=120) == False
        assert poly_rec.check_collision(non_colliding_poly, query_time=120) == False
        assert poly_rec.check_collision(colliding_poly, query_time=130) == True
        assert poly_rec.check_collision(non_colliding_poly, query_time=130) == False
        assert poly_rec.check_collision(colliding_poly, query_time=140) == False
        assert poly_rec.check_collision(non_colliding_poly, query_time=140) == False

        assert (
            poly_rec.check_collision(colliding_poly, query_interval=Interval(120, 123))
            == False
        )
        assert (
            poly_rec.check_collision(
                non_colliding_poly, query_interval=Interval(120, 123)
            )
            == False
        )
        assert (
            poly_rec.check_collision(colliding_poly, query_interval=Interval(125, 130))
            == True
        )
        assert (
            poly_rec.check_collision(
                non_colliding_poly, query_interval=Interval(125, 130)
            )
            == False
        )
        assert (
            poly_rec.check_collision(colliding_poly, query_interval=Interval(130, 140))
            == True
        )
        assert (
            poly_rec.check_collision(
                non_colliding_poly, query_interval=Interval(130, 140)
            )
            == False
        )

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
        point = ShapelyPoint(0.5, 0.5)
        assert polygon.check_collision(point, query_time=5) == True
        assert polygon.check_collision(point, query_interval=in1) == True
        assert polygon.check_collision(point, query_time=15) == False
        assert polygon.check_collision(point, query_interval=in3) == False

        # collision check with line and different temporal queries
        line = ShapelyLine([(0, 0), (1, 1), (2, 2)])
        assert polygon.check_collision(line, query_time=5) == True
        assert polygon.check_collision(line, query_interval=in1) == True
        assert polygon.check_collision(line, query_time=15) == False
        assert polygon.check_collision(line, query_interval=in3) == False

    def test_check_collision_without_time_interval(self):
        polygon = Polygon(geometry=ShapelyPolygon([(0, 0), (1, 1), (2, 2)]), radius=1.0)

        # collision check with point outside
        point = ShapelyPoint(3, 3)
        assert polygon.check_collision(point) == False

        # collision check with point inside
        point = ShapelyPoint(0.5, 0.5)
        assert polygon.check_collision(point) == True

        # collision check with point inside at arbitrary time
        point = ShapelyPoint(0.5, 0.5)
        assert polygon.check_collision(point, query_time=5) == True

        # collision check with point outside at arbitrary time
        point = ShapelyPoint(3, 3)
        assert polygon.check_collision(point, query_time=5) == False

        # collision check with line inside
        line = ShapelyLine([(0, 0), (1, 1), (2, 2)])
        assert polygon.check_collision(line) == True

        # collision check with line outside
        line = ShapelyLine([(4, 4), (3, 3)])
        assert polygon.check_collision(line) == False

        # collision check with line inside at arbitrary time
        line = ShapelyLine([(0, 0), (1, 1), (2, 2)])
        assert polygon.check_collision(line, query_time=5) == True

        # collision check with line outside at arbitrary time
        line = ShapelyLine([(4, 4), (3, 3)])
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
        polygon = Polygon(geometry=ShapelyPolygon([(0, 0), (1, 1), (1, 0)]))
        polygon.plot()  # Plot the polygon on a new figure

        # Test case 2: Plotting on an existing figure
        polygon = Polygon(geometry=ShapelyPolygon([(0, 0), (1, 1), (1, 0)]))
        polygon.plot(fig=fig)  # Plot the polygon on the provided figure

        # Test case 3: Plotting with temporal input arguments
        polygon = Polygon(geometry=ShapelyPolygon([(0, 0), (1, 1), (1, 0)]), radius=1.0)
        polygon.plot(
            fig=fig, query_time=5
        )  # Plot the polygon at a specific time on the provided figure

        # Test case 4: Plotting with temporal input arguments
        polygon = Polygon(geometry=ShapelyPolygon([(0, 0), (1, 1), (1, 0)]), radius=1.0)
        polygon.plot(fig=fig, query_interval=Interval(0, 10))

    def test_load_save(self):
        radius = 5.5
        polygon = ShapelyPolygon([(0, 0), (1, 1), (1, 0)])
        time_interval = Interval(0, 10, closed="left")

        # Test case 1: Convert polygon object to JSON and back (only geometry)
        poly_1 = Polygon(geometry=polygon)
        json_1 = poly_1.export_to_json()
        loaded_1 = Polygon(json_data=json_1)
        assert loaded_1.geometry == polygon
        assert loaded_1.time_interval == None
        assert loaded_1.radius == 0

        # Test case 2: Convert polygon object to JSON and back (only geometry and time interval)
        poly_2 = Polygon(geometry=polygon, time_interval=time_interval)
        json_2 = poly_2.export_to_json()
        loaded_2 = Polygon(json_data=json_2)
        assert loaded_2.geometry == polygon
        assert loaded_2.time_interval == time_interval
        assert loaded_2.radius == 0

        # Test case 3: Convert polygon object to JSON and back (only geometry, time interval, and radius)
        poly_3 = Polygon(geometry=polygon, time_interval=time_interval, radius=radius)
        json_3 = poly_3.export_to_json()
        loaded_3 = Polygon(json_data=json_3)
        assert loaded_3.geometry == polygon
        assert loaded_3.time_interval == time_interval
        assert loaded_3.radius == radius

        # Test case 4: Convert polygon object to JSON and back (only time interval and radius)
        poly_4 = Polygon(time_interval=time_interval, radius=radius)
        json_4 = poly_4.export_to_json()
        loaded_4 = Polygon(json_data=json_4)
        assert loaded_4.geometry == None
        assert loaded_4.time_interval == time_interval
        assert loaded_4.radius == radius

        # Test case 5: Convert polygon object to JSON and back (only geometry and radius)
        poly_5 = Polygon(geometry=polygon, radius=radius)
        json_5 = poly_5.export_to_json()
        loaded_5 = Polygon(json_data=json_5)
        assert loaded_5.geometry == polygon
        assert loaded_5.time_interval == None
        assert loaded_5.radius == radius

        # Test case 6: Convert polygon object to JSON and back (only radius)
        poly_6 = Polygon(radius=radius)
        json_6 = poly_6.export_to_json()
        loaded_6 = Polygon(json_data=json_6)
        assert loaded_6.geometry == None
        assert loaded_6.time_interval == None
        assert loaded_6.radius == radius

        # Test case 7: Convert polygon object to JSON and back (only geometry and time interval)
        poly_7 = Polygon(geometry=polygon, time_interval=time_interval)
        json_7 = poly_7.export_to_json()
        loaded_7 = Polygon(json_data=json_7)
        assert loaded_7.geometry == polygon
        assert loaded_7.time_interval == time_interval
        assert loaded_7.radius == 0

        # Test case 8: Convert polygon object to JSON and back (geometry, time interval, radius, and recurrence)
        poly_8 = Polygon(
            geometry=polygon,
            time_interval=time_interval,
            radius=radius,
            recurrence=Recurrence.MINUTELY,
        )
        json_8 = poly_8.export_to_json()
        loaded_8 = Polygon(json_data=json_8)
        assert loaded_8.geometry == polygon
        assert loaded_8.time_interval == time_interval
        assert loaded_8.radius == radius
        assert loaded_8.recurrence == Recurrence.MINUTELY

        # Test case 9: Convert polygon object to JSON, save and load from file
        poly_8 = Polygon(
            geometry=polygon,
            time_interval=time_interval,
            radius=radius,
            recurrence=Recurrence.HOURLY,
        )
        json_8 = poly_8.export_to_json()

        with open("test_polygon_saving.txt", "w") as f:
            json.dump(json_8, f)

        with open("test_polygon_saving.txt", "r") as f:
            json_obj8_loaded = json.load(f)

        loaded_8 = Polygon(json_data=json_obj8_loaded)
        assert loaded_8.geometry == polygon
        assert loaded_8.time_interval == time_interval
        assert loaded_8.radius == radius
        assert loaded_8.recurrence == Recurrence.HOURLY

        os.remove("test_polygon_saving.txt")

    def test_copy(self):
        polygon = self.setup_method()
        polygon_copy = polygon.copy()

        assert polygon_copy.geometry == polygon.geometry
        assert polygon_copy.time_interval == polygon.time_interval
        assert polygon_copy.radius == polygon.radius
        assert polygon_copy.recurrence == polygon.recurrence

    def test_random(self):
        ## Test random polygon generation with different inputs.
        ## For each combination of inputs, multiple polygons are generated and tested for correct parameters.

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
        max_size = 5

        poly1 = Polygon.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            max_size=max_size,
        )

        coords = poly1.geometry.exterior.coords
        for coord in coords:
            assert coord[0] >= min_x and coord[0] <= max_x
            assert coord[1] >= min_y and coord[1] <= max_y

        assert poly1.time_interval == None or (
            poly1.time_interval.left >= min_interval_default
            and poly1.time_interval.right <= max_interval_default
        )
        assert poly1.radius >= min_radius and poly1.radius <= max_radius
        assert poly1.recurrence == Recurrence.NONE

        min_x = -100
        max_x = 100
        min_y = -100
        max_y = 100
        min_radius = 10
        max_radius = 20

        poly2 = Polygon.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            max_size=max_size,
        )

        coords = poly2.geometry.exterior.coords
        for coord in coords:
            assert coord[0] >= min_x and coord[0] <= max_x
            assert coord[1] >= min_y and coord[1] <= max_y

        assert poly2.time_interval == None or (
            poly2.time_interval.left >= min_interval_default
            and poly2.time_interval.right <= max_interval_default
        )
        assert poly2.radius >= min_radius and poly2.radius <= max_radius
        assert poly2.recurrence == Recurrence.NONE

        # Test case 2 - only static polygons without recurrence
        poly3 = Polygon.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            max_size=max_size,
            only_static=True,
        )

        coords = poly3.geometry.exterior.coords
        for coord in coords:
            assert coord[0] >= min_x and coord[0] <= max_x
            assert coord[1] >= min_y and coord[1] <= max_y

        assert poly3.time_interval == None
        assert poly3.radius >= min_radius and poly3.radius <= max_radius
        assert poly3.recurrence == Recurrence.NONE

        min_x = -200
        max_x = 0
        min_y = -100
        max_y = 100
        min_radius = 20
        max_radius = 30

        poly4 = Polygon.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            max_size=max_size,
            only_static=True,
        )

        coords = poly4.geometry.exterior.coords
        for coord in coords:
            assert coord[0] >= min_x and coord[0] <= max_x
            assert coord[1] >= min_y and coord[1] <= max_y

        assert poly4.time_interval == None
        assert poly4.radius >= min_radius and poly4.radius <= max_radius
        assert poly4.recurrence == Recurrence.NONE

        # Test case 3 - random dynamic polygons with recurrence
        min_x = 0
        max_x = 100
        min_y = 0
        max_y = 100
        min_radius = 0.1
        max_radius = 10

        poly5 = Polygon.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            max_size=max_size,
            only_dynamic=True,
            random_recurrence=True,
        )

        coords = poly5.geometry.exterior.coords
        for coord in coords:
            assert coord[0] >= min_x and coord[0] <= max_x
            assert coord[1] >= min_y and coord[1] <= max_y

        assert poly5.time_interval.left >= min_interval_default
        assert poly5.time_interval.right <= max_interval_default
        assert poly5.radius >= min_radius and poly5.radius <= max_radius
        assert poly5.recurrence in [
            Recurrence.NONE,
            Recurrence.MINUTELY,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]

        poly6 = Polygon.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            max_size=max_size,
            only_dynamic=True,
            random_recurrence=True,
        )

        coords = poly6.geometry.exterior.coords
        for coord in coords:
            assert coord[0] >= min_x and coord[0] <= max_x
            assert coord[1] >= min_y and coord[1] <= max_y

        assert poly6.time_interval.left >= min_interval_default
        assert poly6.time_interval.right <= max_interval_default
        assert poly6.radius >= min_radius and poly6.radius <= max_radius
        assert poly6.recurrence in [
            Recurrence.NONE,
            Recurrence.MINUTELY,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]

        poly7 = Polygon.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            max_size=max_size,
            only_dynamic=True,
            random_recurrence=True,
        )

        coords = poly7.geometry.exterior.coords
        for coord in coords:
            assert coord[0] >= min_x and coord[0] <= max_x
            assert coord[1] >= min_y and coord[1] <= max_y

        assert poly7.time_interval.left >= min_interval_default
        assert poly7.time_interval.right <= max_interval_default
        assert poly7.radius >= min_radius and poly7.radius <= max_radius
        assert poly7.recurrence in [
            Recurrence.NONE,
            Recurrence.MINUTELY,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]

        # Test case 4 - dynamic polygons with custom time interval
        min_interval = 100
        max_interval = 200

        poly8 = Polygon.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            min_interval=min_interval,
            max_interval=max_interval,
            max_size=max_size,
            only_dynamic=True,
            random_recurrence=True,
        )

        coords = poly8.geometry.exterior.coords
        for coord in coords:
            assert coord[0] >= min_x and coord[0] <= max_x
            assert coord[1] >= min_y and coord[1] <= max_y

        assert poly8.time_interval.left >= min_interval
        assert poly8.time_interval.right <= max_interval
        assert poly8.radius >= min_radius and poly8.radius <= max_radius
        assert poly8.recurrence in [
            Recurrence.NONE,
            Recurrence.MINUTELY,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]

        poly9 = Polygon.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            min_interval=min_interval,
            max_interval=max_interval,
            max_size=max_size,
            only_dynamic=True,
            random_recurrence=True,
        )

        coords = poly9.geometry.exterior.coords
        for coord in coords:
            assert coord[0] >= min_x and coord[0] <= max_x
            assert coord[1] >= min_y and coord[1] <= max_y

        assert poly9.time_interval.left >= min_interval
        assert poly9.time_interval.right <= max_interval
        assert poly9.radius >= min_radius and poly9.radius <= max_radius
        assert poly9.recurrence in [
            Recurrence.NONE,
            Recurrence.MINUTELY,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]

        poly10 = Polygon.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            min_interval=min_interval,
            max_interval=max_interval,
            max_size=max_size,
            only_dynamic=True,
        )

        coords = poly10.geometry.exterior.coords
        for coord in coords:
            assert coord[0] >= min_x and coord[0] <= max_x
            assert coord[1] >= min_y and coord[1] <= max_y

        assert poly10.time_interval.left >= min_interval
        assert poly10.time_interval.right <= max_interval
        assert poly10.radius >= min_radius and poly10.radius <= max_radius
        assert poly10.recurrence == Recurrence.NONE

        # Test case 5 - dynamic polygons with custom time interval, radius and random recurrence
        min_x = 1000
        max_x = 2000
        min_y = 3000
        max_y = 4000
        min_radius = 3
        max_radius = 40
        min_interval = 100
        max_interval = 200

        poly11 = Polygon.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            min_interval=min_interval,
            max_interval=max_interval,
            max_size=max_size,
            only_dynamic=True,
            random_recurrence=True,
        )

        coords = poly11.geometry.exterior.coords
        for coord in coords:
            assert coord[0] >= min_x and coord[0] <= max_x
            assert coord[1] >= min_y and coord[1] <= max_y

        assert poly11.radius >= min_radius and poly11.radius <= max_radius

        # check if polygon is static or not and test accordingly
        if poly11.time_interval == None:
            assert poly11.recurrence == Recurrence.NONE
        else:
            assert poly11.time_interval.left >= min_interval
            assert poly11.time_interval.right <= max_interval
            assert poly11.recurrence in [
                Recurrence.NONE,
                Recurrence.MINUTELY,
                Recurrence.HOURLY,
                Recurrence.DAILY,
            ]

        poly12 = Polygon.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            min_interval=min_interval,
            max_interval=max_interval,
            max_size=max_size,
            random_recurrence=True,
        )

        coords = poly12.geometry.exterior.coords
        for coord in coords:
            assert coord[0] >= min_x and coord[0] <= max_x
            assert coord[1] >= min_y and coord[1] <= max_y

        assert poly12.radius >= min_radius and poly12.radius <= max_radius

        # check if polygon is static or not and test accordingly
        if poly12.time_interval == None:
            assert poly12.recurrence == Recurrence.NONE
        else:
            assert poly12.time_interval.left >= min_interval
            assert poly12.time_interval.right <= max_interval
            assert poly12.recurrence in [
                Recurrence.NONE,
                Recurrence.MINUTELY,
                Recurrence.HOURLY,
                Recurrence.DAILY,
            ]

        poly13 = Polygon.random(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_radius=min_radius,
            max_radius=max_radius,
            min_interval=min_interval,
            max_interval=max_interval,
            max_size=max_size,
            random_recurrence=True,
        )

        coords = poly13.geometry.exterior.coords
        for coord in coords:
            assert coord[0] >= min_x and coord[0] <= max_x
            assert coord[1] >= min_y and coord[1] <= max_y

        assert poly13.radius >= min_radius and poly13.radius <= max_radius

        # check if polygon is static or not and test accordingly
        if poly13.time_interval == None:
            assert poly13.recurrence == Recurrence.NONE
        else:
            assert poly13.time_interval.left >= min_interval
            assert poly13.time_interval.right <= max_interval
            assert poly13.recurrence in [
                Recurrence.NONE,
                Recurrence.MINUTELY,
                Recurrence.HOURLY,
                Recurrence.DAILY,
            ]
