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
