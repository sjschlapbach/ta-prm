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

    def get_seconds(self) -> int:
        """
        Returns the number of seconds in the recurrence interval.

        Returns
        -------
        int
            the number of seconds in the recurrence interval
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
