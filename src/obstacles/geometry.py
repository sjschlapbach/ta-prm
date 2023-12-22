from shapely.geometry import Point, LineString, Polygon
from pandas import Interval


class Geometry:
    """
    Represents a geometric object with a radius and time interval.

    Attributes:
        radius (float): The radius around the point, considered to be in collision.
        time_interval (Interval): The closed time interval.

    Methods:
        __init__(radius: float = None, interval: Interval = None): Initializes a Geometry object with the given radius and time interval.
        set_interval(lower_bound: float, upper_bound: float): Sets the closed time interval from a lower and upper bound.
        set_radius(radius: float): Sets the radius around the point, considered to be in collision.
        is_active(query_time: float = None, query_interval: Interval = None): Checks if the geometry is active at a given time or time interval.
        export_to_json(): Returns a JSON representation of the geometry.
        load_from_json(json_object: dict): Loads radius and time interval from a JSON object.
        interval_from_string(input_str: str): Creates a pandas interval from a string.
    """

    def __init__(self, radius: float = None, interval: Interval = None):
        """
        Initializes a Geometry object with the given radius and time interval.

        Args:
            radius (float): The radius around the point, considered to be in collision.
            interval (Interval): The closed time interval.
        """
        self.radius = radius
        self.time_interval = interval

    def set_interval(self, lower_bound: float, upper_bound: float):
        """
        Sets the closed time interval from a lower and upper bound.

        Args:
            lower_bound (float): The lower bound of the interval.
            upper_bound (float): The upper bound of the interval.
        """
        if lower_bound > upper_bound:
            raise ValueError("The lower bound must be smaller than the upper bound.")

        self.time_interval = Interval(lower_bound, upper_bound, closed="both")

    def set_radius(self, radius: float):
        """
        Sets the radius around the point, considered to be in collision.

        Args:
            radius (float): The radius around the point.
        """
        self.radius = radius

    def is_active(self, query_time: float = None, query_interval: Interval = None):
        """
        Checks if the geometry is active at a given time or time interval.

        Args:
            query_time (optional): The specific time to check activity at.
            query_interval (optional): The time interval to check activity within.

        Returns:
            bool: True if active, False otherwise. Objects without a time interval are always active.
        """
        if self.time_interval is None:
            return True

        if query_time is not None:
            return query_time in self.time_interval

        if query_interval is not None:
            return self.time_interval.overlaps(query_interval)

        return True

    def export_to_json(self):
        """
        Returns a JSON representation of the geometry.

        Returns:
            str: A JSON representation of the geometry.
        """
        return {"radius": str(self.radius), "interval": str(self.time_interval)}

    def load_from_json(self, json_object: dict):
        """
        Loads radius and time interval from a JSON object.

        Args:
            json_object (dict): The JSON object to load from.
        """
        # extract radius and interval from json object
        radius_str = json_object["radius"]
        interval_str = json_object["interval"]

        # convert radius and interval to float and Interval
        self.radius = float(radius_str) if radius_str != "None" else None
        self.time_interval = self.interval_from_string(interval_str)

    def interval_from_string(self, input_str: str):
        """
        Creates a pandas interval from a string.

        Args:
            input_str (str): The string to create the interval from.

        Returns:
            Interval: The pandas interval.
        """
        if input_str == "None":
            return None

        if len(input_str) < 2:
            raise ValueError("Invalid interval string with length less than 2.")

        interval_closed = "both"
        if input_str[0] == "(" and input_str[-1] == ")":
            interval_closed = "neither"
        elif input_str[0] == "(":
            interval_closed = "right"
        elif input_str[-1] == ")":
            interval_closed = "left"

        bounds = input_str[1:-1].split(",")
        return Interval(
            float(bounds[0].strip()), float(bounds[1].strip()), closed=interval_closed
        )
