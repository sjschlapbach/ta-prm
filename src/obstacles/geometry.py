from shapely.geometry import Point, LineString, Polygon
from pandas import Interval

from src.util.recurrence import Recurrence


class Geometry:
    """
    Represents a geometric object with a radius, time interval, and recurrence.

    Attributes:
        radius (float): The radius around the point, considered to be in collision.
        time_interval (Interval): The closed time interval.
        recurrence (Recurrence): The recurrence frequency.

    Methods:
        __init__(radius: float = None, interval: Interval = None, recurrence: Recurrence = None, json_data: dict = None): Initializes a Geometry object with the given radius, time interval, recurrence, and json_data.
        set_interval(lower_bound: float, upper_bound: float): Sets the closed time interval from a lower and upper bound.
        set_radius(radius: float): Sets the radius around the point, considered to be in collision.
        set_recurrence(recurrence: Recurrence): Sets the recurrence frequency.
        is_active(query_time: float = None, query_interval: Interval = None): Checks if the geometry is active at a given time or time interval.
        export_to_json(): Returns a JSON representation of the geometry.
        load_from_json(json_object: dict): Loads radius and time interval from a JSON object.
        interval_from_string(input_str: str): Creates a pandas interval from a string.
    """

    def __init__(
        self,
        radius: float = None,
        interval: Interval = None,
        recurrence: Recurrence = None,
        json_data: dict = None,
    ):
        """
        Initializes a Geometry object with the given radius, time interval, and recurrence.

        Args:
            radius (float, optional): The radius around the point, considered to be in collision.
            interval (Interval, optional): The closed time interval.
            recurrence (Recurrence, optional): The recurrence frequency.
            json_data (dict, optional): A dictionary containing serialized data to load from.

        If `json_data` is provided, the object will be initialized by loading data from it.
        Otherwise, the object will be initialized with the provided `radius`, `interval`, and `recurrence`.
        If `recurrence` is not provided, it defaults to `Recurrence.NONE`.
        """
        if json_data is not None:
            self.load_from_json(json_data)
            return

        self.radius = radius
        self.time_interval = interval

        if recurrence is None:
            self.recurrence = Recurrence.NONE
        else:
            self.recurrence = recurrence

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

    def set_recurrence(self, recurrence: Recurrence):
        """
        Sets the recurrence frequency.

        Args:
            recurrence (Recurrence): The recurrence frequency.
        """
        self.recurrence = recurrence

    def is_active(self, query_time: float = None, query_interval: Interval = None):
        """
        Checks if the geometry is active at a given time or time interval.

        Args:
            query_time (optional): The specific time to check activity at.
            query_interval (optional): The time interval to check activity within.

        Returns:
            bool: True if active, False otherwise. Objects without a time interval are always active.
        """
        # if neither query_time nor query_interval is given, consider the obstacle to be active
        if query_time is None and query_interval is None:
            return True

        # obstacles without time interval are considered to be static / active all the time
        if self.time_interval is None:
            return True
        else:
            # check if time interval overlaps with query time or interval in the case of no recurrence
            if self.recurrence == Recurrence.NONE:
                if query_time is not None:
                    return query_time in self.time_interval
                elif query_interval is not None:
                    return self.time_interval.overlaps(query_interval)

                # if neither query_time nor query_interval is given, consider the obstacle to be active
                return True

            # check if time interval overlaps with query time or interval in the case of recurrence
            if query_time is not None:
                # if query_time is before obstacle_start, return false
                if query_time < self.time_interval.left:
                    return False

                # find the occurence of the obstacle, which includes the query_time
                delta = query_time - self.time_interval.left
                recurrence_length = self.recurrence.get_seconds()
                occurence = delta // recurrence_length

                # check if query_time is in the time interval of the occurence
                if query_time in self.time_interval + occurence * recurrence_length:
                    return True
                else:
                    return False

            else:
                # if query_time is before obstacle_start, return false
                if query_interval.right < self.time_interval.left:
                    return False

                # find the occurence covered by the query_interval
                delta_start = query_interval.left - self.time_interval.left
                delta_end = query_interval.right - self.time_interval.left
                recurrence_length = self.recurrence.get_seconds()
                start_k = delta_start // recurrence_length
                end_k = delta_end // recurrence_length

                # if the query interval spans over multiple occurence, at least one of the inner ones must be active
                if end_k - start_k > 0:
                    return True
                # if start and end are equal, check the corresponding occurence for activity
                elif start_k == end_k:
                    return query_interval.overlaps(
                        self.time_interval + start_k * recurrence_length
                    )
                # other cases should not occur, throw an error
                else:
                    raise ValueError(
                        "There occurred an error while checking for activity with recurrence parameter. Computed start and end parameters: "
                        + str(start_k)
                        + ", "
                        + str(end_k)
                    )

        raise ValueError("There occurred an error while checking for activity.")

    def export_to_json(self):
        """
        Returns a JSON representation of the geometry.

        Returns:
            str: A JSON representation of the geometry.
        """
        return {
            "radius": str(self.radius),
            "interval": str(self.time_interval),
            "recurrence": self.recurrence.to_string(),
        }

    def load_from_json(self, json_object: dict):
        """
        Loads radius and time interval from a JSON object.

        Args:
            json_object (dict): The JSON object to load from.
        """
        # extract radius and interval from json object
        radius_str = json_object["radius"]
        interval_str = json_object["interval"]
        recurrence_str = json_object["recurrence"]

        # convert stringified values back to python class instances
        self.radius = float(radius_str) if radius_str != "None" else None
        self.time_interval = self.interval_from_string(interval_str)
        self.recurrence = Recurrence.from_string(recurrence_str)

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
