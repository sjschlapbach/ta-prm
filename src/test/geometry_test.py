import pytest
import json
import os
from shapely.geometry import Polygon, Point, LineString
from pandas import Interval

from src.obstacles.geometry import Geometry


class TestGeometry:
    def test_geometry_init(self):
        # Test case with valid inputs
        radius = 5.0
        interval = Interval(0, 10)
        geometry = Geometry(radius, interval)
        assert geometry.radius == radius
        assert geometry.time_interval == interval

        # Test case with None inputs
        geometry = Geometry(None, None)
        assert geometry.radius is None
        assert geometry.time_interval is None

        # Test case with only radius specified
        radius = 3.0
        geometry = Geometry(radius, None)
        assert geometry.radius == radius
        assert geometry.time_interval is None

        # Test case with only interval specified
        interval = Interval(0, 5)
        geometry = Geometry(None, interval)
        assert geometry.radius is None
        assert geometry.time_interval == interval

        # Test case with no inputs specified
        geometry = Geometry(None, None)
        assert geometry.radius is None
        assert geometry.time_interval is None

    def test_set_interval(self):
        geometry = Geometry()

        # Test valid interval
        geometry.set_interval(1.0, 2.0)
        assert geometry.time_interval.left == 1.0
        assert geometry.time_interval.right == 2.0

        # Test invalid interval (lower bound > upper bound)
        with pytest.raises(ValueError):
            geometry.set_interval(3.0, 2.0)

        # Test missing arguments
        with pytest.raises(TypeError):
            geometry.set_interval(1.0)

        with pytest.raises(TypeError):
            geometry.set_interval()

        with pytest.raises(TypeError):
            geometry.set_interval(1.0, 2.0, 3.0)

    def test_set_radius(self):
        # Test case where inputs are provided
        geometry = Geometry()
        geometry.set_radius(5.0)
        assert geometry.radius == 5.0

        # Test case where no arguments are provided
        geometry = Geometry()
        with pytest.raises(TypeError):
            geometry.set_radius()

    def test_is_active(self):
        # Test case 1: Object without a time interval should always be active
        geometry1 = Geometry()
        assert geometry1.is_active() == True

        # Test case 2: Object with a time interval should be active within the interval
        interval = Interval(0, 10)
        geometry2 = Geometry(interval=interval)
        assert geometry2.is_active(5) == True

        # Test case 3: Object with a time interval should not be active outside the interval
        interval = Interval(0, 10)
        geometry3 = Geometry(interval=interval)
        assert geometry3.is_active(15) == False

        # Test case 4: Object with a time interval should overlap with the provided query interval
        interval = Interval(0, 10)
        query_interval = Interval(5, 15)
        geometry4 = Geometry(interval=interval)
        assert geometry4.is_active(query_interval=query_interval) == True

        # Test case 5: Object with a time interval should not overlap with the provided query interval
        interval = Interval(0, 10)
        query_interval = Interval(15, 20)
        geometry5 = Geometry(interval=interval)
        assert geometry5.is_active(query_interval=query_interval) == False

        # Test case 6: Object with a time interval should be active at the provided query time
        interval = Interval(0, 10)
        geometry6 = Geometry(interval=interval)
        assert geometry6.is_active(query_time=5) == True

        # Test case 7: Object with a time interval should not be active at the provided query time
        interval = Interval(0, 10)
        geometry7 = Geometry(interval=interval)
        assert geometry7.is_active(query_time=15) == False

        # Test case 8: Object without a time interval should always be active, even with query time or query interval
        geometry8 = Geometry()
        assert geometry8.is_active(query_time=5) == True
        assert geometry8.is_active(query_interval=Interval(0, 10)) == True

    def test_export_and_load_geometry(self):
        # test parameters
        radius = 2.5
        interval_default = Interval(0.0, 10.0)
        interval_open = Interval(1.5, 11.5, closed="neither")
        interval_closed = Interval(3, 13.0, closed="both")
        interval_left = Interval(5, 15.0, closed="left")
        interval_right = Interval(7, 17.0, closed="right")

        # Test case 1: Export and load geometry with radius and no interval
        geometry = Geometry(radius=radius)
        json_obj1 = geometry.export_to_json()
        loaded_geometry1 = Geometry()
        loaded_geometry1.load_from_json(json_obj1)
        assert loaded_geometry1.radius == radius

        # Test case 2: Export and load geometry with default interval and no radius
        geometry = Geometry(interval=interval_default)
        json_obj2 = geometry.export_to_json()
        loaded_geometry2 = Geometry()
        loaded_geometry2.load_from_json(json_obj2)
        assert loaded_geometry2.time_interval == interval_default

        # Test case 3: Export and load geometry with default interval and radius
        geometry = Geometry(radius=radius, interval=interval_default)
        json_obj3 = geometry.export_to_json()
        loaded_geometry3 = Geometry()
        loaded_geometry3.load_from_json(json_obj3)
        assert loaded_geometry3.radius == radius
        assert loaded_geometry3.time_interval == interval_default

        # Test case 4: Export and load geometry with open interval and radius
        geometry = Geometry(radius=radius, interval=interval_open)
        json_obj4 = geometry.export_to_json()
        loaded_geometry4 = Geometry()
        loaded_geometry4.load_from_json(json_obj4)
        assert loaded_geometry4.radius == radius
        assert loaded_geometry4.time_interval == interval_open

        # Test case 5: Export and load geometry with closed interval and radius
        geometry = Geometry(radius=radius, interval=interval_closed)
        json_obj5 = geometry.export_to_json()
        loaded_geometry5 = Geometry()
        loaded_geometry5.load_from_json(json_obj5)
        assert loaded_geometry5.radius == radius
        assert loaded_geometry5.time_interval == interval_closed

        # Test case 6: Export and load geometry with left interval and radius
        geometry = Geometry(radius=radius, interval=interval_left)
        json_obj6 = geometry.export_to_json()
        loaded_geometry6 = Geometry()
        loaded_geometry6.load_from_json(json_obj6)
        assert loaded_geometry6.radius == radius
        assert loaded_geometry6.time_interval == interval_left

        # Test case 7: Export and load geometry with right interval and radius
        geometry = Geometry(radius=radius, interval=interval_right)
        json_obj7 = geometry.export_to_json()
        loaded_geometry7 = Geometry()
        loaded_geometry7.load_from_json(json_obj7)
        assert loaded_geometry7.radius == radius
        assert loaded_geometry7.time_interval == interval_right

        # Test case 8: Save and load geometry from file
        geometry = Geometry(radius=radius, interval=interval_right)
        json_obj8 = geometry.export_to_json()

        with open("test_geometry_saving.txt", "w") as f:
            json.dump(json_obj8, f)

        with open("test_geometry_saving.txt", "r") as f:
            json_obj8_loaded = json.load(f)

        loaded_geometry8 = Geometry()
        loaded_geometry8.load_from_json(json_obj8_loaded)
        assert loaded_geometry8.radius == radius
        assert loaded_geometry8.time_interval == interval_right

        os.remove("test_geometry_saving.txt")
