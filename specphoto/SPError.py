## Error class for package

class SPException(Exception):
    """Base class for exceptions in this module."""
    pass

class SPError(SPException):
    """Exception raised for general errors.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
