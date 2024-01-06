import pytest
import json
import os
from shapely.geometry import Polygon, Point, LineString
from pandas import Interval

from src.obstacles.geometry import Geometry
from src.util.recurrence import Recurrence


class TestGeometry:
    def test_geometry_init(self):
        # Test case with valid inputs
        radius = 5.0
        interval = Interval(0, 10)
        geometry = Geometry(radius, interval, Recurrence.MINUTELY)
        assert geometry.radius == radius
        assert geometry.time_interval == interval
        assert geometry.recurrence == Recurrence.MINUTELY

        # Test case with None inputs
        geometry = Geometry(None, None)
        assert geometry.radius is None
        assert geometry.time_interval is None
        assert geometry.recurrence == Recurrence.NONE

        # Test case with only radius specified
        radius = 3.0
        geometry = Geometry(radius, None)
        assert geometry.radius == radius
        assert geometry.time_interval is None
        assert geometry.recurrence == Recurrence.NONE

        # Test case with only interval specified
        interval = Interval(0, 5)
        geometry = Geometry(None, interval)
        assert geometry.radius is None
        assert geometry.time_interval == interval
        assert geometry.recurrence == Recurrence.NONE

        # Test case with only recurrence specified
        geometry = Geometry(None, None, Recurrence.HOURLY)
        assert geometry.radius is None
        assert geometry.time_interval is None
        assert geometry.recurrence == Recurrence.HOURLY

        # Test case with no inputs specified
        geometry = Geometry(None, None)
        assert geometry.radius is None
        assert geometry.time_interval is None
        assert geometry.recurrence == Recurrence.NONE

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

    def test_set_recurrence(self):
        # Test cases where inputs are provided
        geometry = Geometry()
        geometry.set_recurrence(Recurrence.NONE)
        assert geometry.recurrence == Recurrence.NONE
        geometry.set_recurrence(Recurrence.MINUTELY)
        assert geometry.recurrence == Recurrence.MINUTELY
        geometry.set_recurrence(Recurrence.HOURLY)
        assert geometry.recurrence == Recurrence.HOURLY
        geometry.set_recurrence(Recurrence.DAILY)
        assert geometry.recurrence == Recurrence.DAILY

        # Test case where no arguments are provided
        geometry = Geometry()
        with pytest.raises(TypeError):
            geometry.set_recurrence()

    def test_is_active_no_recurrence(self):
        ## TEST CASES WITHOUT RECURRENCE
        # Test case 1: Object without a time interval should always be active
        geometry1 = Geometry(recurrence=Recurrence.NONE)
        assert geometry1.is_active() == True
        geometry1 = Geometry(recurrence=Recurrence.MINUTELY)
        assert geometry1.is_active() == True
        geometry1 = Geometry(recurrence=Recurrence.HOURLY)
        assert geometry1.is_active() == True
        geometry1 = Geometry(recurrence=Recurrence.DAILY)
        assert geometry1.is_active() == True

        # Test case 2: Object with a time interval should be active within the interval
        interval = Interval(0, 10)
        geometry2 = Geometry(interval=interval, recurrence=Recurrence.NONE)
        assert geometry2.is_active(query_time=5) == True

        # Test case 3: Object with a time interval should not be active outside the interval
        interval = Interval(0, 10)
        geometry3 = Geometry(interval=interval)
        assert geometry3.is_active(query_time=15) == False

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

    def test_is_active_min_recurrence(self):
        ## TEST CASES WITH MINUTELY RECURRENCE
        # Test case 1: Query interval before first obstacle recurrence should always be inactive
        geom_in = Interval(10, 20)
        geometry_min = Geometry(interval=geom_in, recurrence=Recurrence.MINUTELY)
        assert geometry_min.is_active(query_time=5) == False
        assert geometry_min.is_active(query_interval=Interval(0, 5)) == False

        # Test case 2: Query interval intersecting with first occurence should be active
        assert geometry_min.is_active(query_time=15) == True
        assert geometry_min.is_active(query_interval=Interval(5, 15)) == True

        # Test case 3: Query interval inside first occurence should be active
        assert geometry_min.is_active(query_interval=Interval(15, 17)) == True

        # Test case 4: Query interval intersecting end of first occurence should be active
        assert geometry_min.is_active(query_interval=Interval(17, 25)) == True

        # Test case 5: Query interval after first occurence should be inactive
        assert geometry_min.is_active(query_time=25) == False
        assert geometry_min.is_active(query_interval=Interval(25, 30)) == False

        # Test case 6: Query interval covering entire first occurence should be active
        assert geometry_min.is_active(query_interval=Interval(5, 25)) == True

        # Test case 7: Query interval intersecting start of future occurence should be active
        assert geometry_min.is_active(query_time=125) == False
        assert geometry_min.is_active(query_interval=Interval(125, 135)) == True

        # Test case 8: Query interval inside future occurence should be active
        assert geometry_min.is_active(query_time=135) == True
        assert geometry_min.is_active(query_interval=Interval(135, 137)) == True

        # Test case 9: Query interval intersecting end of future occurence should be active
        assert geometry_min.is_active(query_interval=Interval(137, 145)) == True

        # Test case 10: Query interval after future occurence should be inactive
        assert geometry_min.is_active(query_time=145) == False
        assert geometry_min.is_active(query_interval=Interval(145, 150)) == False

        # Test case 11: Query interval covering entire future occurence should be active
        assert geometry_min.is_active(query_interval=Interval(125, 145)) == True

        # Test case 12: Query interval covering multiple occurence should be active
        assert geometry_min.is_active(query_interval=Interval(5, 145)) == True

    def test_is_active_hour_recurrence(self):
        ## TEST CASES WITH HOURLY RECURRENCE
        # Test case 1: Query interval before first obstacle recurrence should always be inactive
        geom_in = Interval(10, 20)
        geometry_hour = Geometry(interval=geom_in, recurrence=Recurrence.HOURLY)
        assert geometry_hour.is_active(query_time=5) == False
        assert geometry_hour.is_active(query_interval=Interval(0, 5)) == False

        # Test case 2: Query interval intersecting with first occurence should be active
        assert geometry_hour.is_active(query_time=15) == True
        assert geometry_hour.is_active(query_interval=Interval(5, 15)) == True

        # Test case 3: Query interval inside first occurence should be active
        assert geometry_hour.is_active(query_interval=Interval(15, 17)) == True

        # Test case 4: Query interval intersecting end of first occurence should be active
        assert geometry_hour.is_active(query_interval=Interval(17, 25)) == True

        # Test case 5: Query interval after first occurence should be inactive
        assert geometry_hour.is_active(query_time=25) == False
        assert geometry_hour.is_active(query_time=125) == False
        assert geometry_hour.is_active(query_time=135) == False
        assert geometry_hour.is_active(query_interval=Interval(25, 30)) == False
        assert geometry_hour.is_active(query_interval=Interval(125, 135)) == False

        # Test case 6: Query interval covering entire first occurence should be active
        assert geometry_hour.is_active(query_interval=Interval(5, 25)) == True

        # Test case 7: Query interval intersecting start of future occurence should be active
        assert geometry_hour.is_active(query_time=7205) == False
        assert geometry_hour.is_active(query_interval=Interval(7205, 7215)) == True

        # Test case 8: Query interval inside future occurence should be active
        assert geometry_hour.is_active(query_time=7215) == True
        assert geometry_hour.is_active(query_interval=Interval(7215, 7217)) == True

        # Test case 9: Query interval intersecting end of future occurence should be active
        assert geometry_hour.is_active(query_interval=Interval(7217, 7225)) == True

        # Test case 10: Query interval after future occurence should be inactive
        assert geometry_hour.is_active(query_time=7225) == False
        assert geometry_hour.is_active(query_interval=Interval(7225, 7230)) == False

        # Test case 11: Query interval covering entire future occurence should be active
        assert geometry_hour.is_active(query_interval=Interval(7205, 7225)) == True

        # Test case 12: Query interval covering multiple occurence should be active
        assert geometry_hour.is_active(query_interval=Interval(5, 7225)) == True

    def test_is_active_day_recurrence(self):
        ## TEST CASES WITH DAILY RECURRENCE
        # Test case 1: Query interval before first obstacle recurrence should always be inactive
        geom_in = Interval(10, 20)
        geometry_day = Geometry(interval=geom_in, recurrence=Recurrence.DAILY)
        assert geometry_day.is_active(query_time=5) == False
        assert geometry_day.is_active(query_interval=Interval(0, 5)) == False

        # Test case 2: Query interval intersecting with first occurence should be active
        assert geometry_day.is_active(query_time=15) == True
        assert geometry_day.is_active(query_interval=Interval(5, 15)) == True

        # Test case 3: Query interval inside first occurence should be active
        assert geometry_day.is_active(query_interval=Interval(15, 17)) == True

        # Test case 4: Query interval intersecting end of first occurence should be active
        assert geometry_day.is_active(query_interval=Interval(17, 25)) == True

        # Test case 5: Query interval after first occurence should be inactive
        assert geometry_day.is_active(query_time=25) == False
        assert geometry_day.is_active(query_time=125) == False
        assert geometry_day.is_active(query_time=135) == False
        assert geometry_day.is_active(query_time=7205) == False
        assert geometry_day.is_active(query_time=7215) == False
        assert geometry_day.is_active(query_interval=Interval(25, 30)) == False
        assert geometry_day.is_active(query_interval=Interval(125, 135)) == False
        assert geometry_day.is_active(query_interval=Interval(7205, 7215)) == False

        # Test case 6: Query interval covering entire first occurence should be active
        assert geometry_day.is_active(query_interval=Interval(5, 25)) == True

        # Test case 7: Query interval intersecting start of future occurence should be active
        assert geometry_day.is_active(query_time=172805) == False
        assert geometry_day.is_active(query_interval=Interval(172805, 172815)) == True

        # Test case 8: Query interval inside future occurence should be active
        assert geometry_day.is_active(query_time=172815) == True
        assert geometry_day.is_active(query_interval=Interval(172815, 172817)) == True

        # Test case 9: Query interval intersecting end of future occurence should be active
        assert geometry_day.is_active(query_interval=Interval(172817, 172825)) == True

        # Test case 10: Query interval after future occurence should be inactive
        assert geometry_day.is_active(query_time=172825) == False
        assert geometry_day.is_active(query_interval=Interval(172825, 172830)) == False

        # Test case 11: Query interval covering entire future occurence should be active
        assert geometry_day.is_active(query_interval=Interval(172805, 172825)) == True

        # Test case 12: Query interval covering multiple occurence should be active
        assert geometry_day.is_active(query_interval=Interval(5, 172825)) == True

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
        geometry = Geometry(
            radius=radius, interval=interval_right, recurrence=Recurrence.DAILY
        )
        json_obj8 = geometry.export_to_json()

        with open("test_geometry_saving.txt", "w") as f:
            json.dump(json_obj8, f)

        with open("test_geometry_saving.txt", "r") as f:
            json_obj8_loaded = json.load(f)

        loaded_geometry8 = Geometry()
        loaded_geometry8.load_from_json(json_obj8_loaded)
        assert loaded_geometry8.radius == radius
        assert loaded_geometry8.time_interval == interval_right
        assert loaded_geometry8.recurrence == Recurrence.DAILY

        os.remove("test_geometry_saving.txt")
