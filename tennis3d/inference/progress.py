"""Optional progress callbacks shared by the CLI and local application."""


class Cancelled(Exception):
    """The caller requested cancellation before the next unit of work."""


def report(callback, stage, completed=0, total=None, unit="frames"):
    if callback is not None:
        callback(stage, completed, total, unit)
