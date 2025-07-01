from loguru import logger
import time

logger.remove()

logger.add(
    "petroscope.log",
    # format="<green>{time}</green> <level>{level}</level> <cyan>{message}</cyan>",
    rotation="10 MB",
    level="DEBUG"
)


def log_time(label, logger):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.debug(f"{label} {elapsed:.4f}")
            return result
        return wrapper
    return decorator
