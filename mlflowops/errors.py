class ServiceError(Exception):
    """Error that is safe to show to API clients."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


class NotFound(ServiceError):
    def __init__(self, message: str):
        super().__init__(message, 404)
