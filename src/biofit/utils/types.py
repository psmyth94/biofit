class Unset:
    """A class to represent an unset value.

    This is used to differentiate between a value that is not set and a value that is set to None.
    Optionally, a description can be provided to indicate what the actual default is when Unset is used.
    """

    def __init__(self, description: str = ""):
        self.description = description

    def __repr__(self) -> str:
        """Return the string representation of the class, including any default description if provided.

        Returns:
            The string representation of the class.
        """
        if self.description:
            return f"Unset (default: {self.description})"
        return "Unset"

    def __bool__(self) -> bool:
        """Return False when the class is used in a boolean context.

        Returns:
            False
        """
        return False
