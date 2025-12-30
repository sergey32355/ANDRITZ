class UndefinedException(Exception):
    """
    TBC
    """

    def __init__(self, message: str = "Undefined Exception.") -> None:
        """
        TBC
        """
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        """
        TBC
        """
        return self.message
