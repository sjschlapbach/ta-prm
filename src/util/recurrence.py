from enum import Enum


class Recurrence(Enum):
    """
    Enum representing different recurrence options.
    """

    # No recurrence.
    NONE = 0

    # Recurs every minute. (module 60 seconds)
    MINUTELY = 1

    # Recurs every hour. (modulo 3600 seconds)
    HOURLY = 2

    # Recurs every day. (modulo 86400 seconds)
    DAILY = 3

    def get_seconds(self):
        """
        Returns the number of seconds corresponding to the recurrence.
        """
        if self == Recurrence.NONE:
            return 0
        elif self == Recurrence.MINUTELY:
            return 60
        elif self == Recurrence.HOURLY:
            return 3600
        elif self == Recurrence.DAILY:
            return 86400
        else:
            raise ValueError("Invalid recurrence value.")

    def to_string(self):
        """
        Returns a string representation of the recurrence.
        """
        if self == Recurrence.NONE:
            return "none"
        elif self == Recurrence.MINUTELY:
            return "minutely"
        elif self == Recurrence.HOURLY:
            return "hourly"
        elif self == Recurrence.DAILY:
            return "daily"
        else:
            raise ValueError("Invalid recurrence value.")

    def from_string(recurrence_str: str):
        """
        Creates a recurrence from a string.
        """
        if recurrence_str == "none":
            return Recurrence.NONE
        elif recurrence_str == "minutely":
            return Recurrence.MINUTELY
        elif recurrence_str == "hourly":
            return Recurrence.HOURLY
        elif recurrence_str == "daily":
            return Recurrence.DAILY
        else:
            raise ValueError("Invalid recurrence string.")
