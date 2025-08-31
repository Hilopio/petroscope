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

            elapsed_hours = elapsed // 3600
            elapsed = elapsed % 3600
            elapsed_minutes = elapsed // 60
            elapsed = elapsed % 60
            if elapsed_hours > 0:
                elapsed_message = f"{elapsed_hours}h {elapsed_minutes}m {elapsed:.2f}s"
            elif elapsed_minutes > 0:
                elapsed_message = f"{elapsed_minutes}m {elapsed:.2f}s"
            else:
                elapsed_message = f"{elapsed:.2f}s"

            logger.debug(f"{label} {elapsed_message:.4f}")
            return result
        return wrapper
    return decorator
