import pytest
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
