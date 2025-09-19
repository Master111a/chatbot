from .prompt_utils import PromptUtils

__all__ = ["PromptUtils"]


from .time_handler import CustomDateTime

def now():
    return CustomDateTime.now()


def set_timezone(tz_name: str):
    CustomDateTime.set_timezone(tz_name)


def fromtimestamp(timestamp: float):
    return CustomDateTime.fromtimestamp(timestamp)
