import pytest

from src.util.recurrence import Recurrence


class TestGeometry:
    def test_get_seconds(self):
        rec_none = Recurrence.NONE
        rec_min = Recurrence.MINUTELY
        rec_hour = Recurrence.HOURLY
        rec_day = Recurrence.DAILY

        assert rec_none.get_seconds() == 0
        assert rec_min.get_seconds() == 60
        assert rec_hour.get_seconds() == 3600
        assert rec_day.get_seconds() == 86400

    def test_recurrence_stringify(self):
        rec_none = Recurrence.NONE
        rec_min = Recurrence.MINUTELY
        rec_hour = Recurrence.HOURLY
        rec_day = Recurrence.DAILY

        assert rec_none.to_string() == "none"
        assert rec_min.to_string() == "minutely"
        assert rec_hour.to_string() == "hourly"
        assert rec_day.to_string() == "daily"

    def test_recurrence_from_string(self):
        rec_none = Recurrence.NONE
        rec_min = Recurrence.MINUTELY
        rec_hour = Recurrence.HOURLY
        rec_day = Recurrence.DAILY

        assert Recurrence.from_string("none") == rec_none
        assert Recurrence.from_string("minutely") == rec_min
        assert Recurrence.from_string("hourly") == rec_hour
        assert Recurrence.from_string("daily") == rec_day

    def test_recurrence_random(self):
        # test random recurrence without minimum duration
        assert Recurrence.random() in [
            Recurrence.NONE,
            Recurrence.MINUTELY,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]
        assert Recurrence.random() in [
            Recurrence.NONE,
            Recurrence.MINUTELY,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]
        assert Recurrence.random() in [
            Recurrence.NONE,
            Recurrence.MINUTELY,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]

        # test random recurrence with minimum duration below one minute
        assert Recurrence.random(min_duration=0) in [
            Recurrence.NONE,
            Recurrence.MINUTELY,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]
        assert Recurrence.random(min_duration=20) in [
            Recurrence.NONE,
            Recurrence.MINUTELY,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]
        assert Recurrence.random(min_duration=40) in [
            Recurrence.NONE,
            Recurrence.MINUTELY,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]
        assert Recurrence.random(min_duration=59) in [
            Recurrence.NONE,
            Recurrence.MINUTELY,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]

        # test random recurrence with minimum duration below one hour
        assert Recurrence.random(min_duration=60) in [
            Recurrence.NONE,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]
        assert Recurrence.random(min_duration=60) in [
            Recurrence.NONE,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]
        assert Recurrence.random(min_duration=120) in [
            Recurrence.NONE,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]
        assert Recurrence.random(min_duration=240) in [
            Recurrence.NONE,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]
        assert Recurrence.random(min_duration=3599) in [
            Recurrence.NONE,
            Recurrence.HOURLY,
            Recurrence.DAILY,
        ]

        # test random recurrence with minimum duration below one day
        assert Recurrence.random(min_duration=3600) in [
            Recurrence.NONE,
            Recurrence.DAILY,
        ]
        assert Recurrence.random(min_duration=3600) in [
            Recurrence.NONE,
            Recurrence.DAILY,
        ]
        assert Recurrence.random(min_duration=7200) in [
            Recurrence.NONE,
            Recurrence.DAILY,
        ]
        assert Recurrence.random(min_duration=14400) in [
            Recurrence.NONE,
            Recurrence.DAILY,
        ]
        assert Recurrence.random(min_duration=86399) in [
            Recurrence.NONE,
            Recurrence.DAILY,
        ]

        # test random recurrence with minimum duration above one day
        assert Recurrence.random(min_duration=86400) == Recurrence.NONE
        assert Recurrence.random(min_duration=86400) == Recurrence.NONE
        assert Recurrence.random(min_duration=172800) == Recurrence.NONE
        assert Recurrence.random(min_duration=345600) == Recurrence.NONE
