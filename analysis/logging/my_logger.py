import sys

from loguru import logger

### ONLY SHOW WARNING AND ABOVE LEVEL MESSAGES ###

logger.remove()  # Remove the default logger
logger.add(sys.stdout, level="WARNING")  # Add a new logger that only shows WARNING and above level messages

###
# As an alternative you can set the LOGURU_LEVEL environment variable to WARNING before running the script.
# Example:
#   export LOGURU_LEVEL=WARNING
#   uvr python your_script.py
###

logger.debug("Used for debugging your code.")  # This will not be shown due to WARNING level
logger.info("Informative messages from your code.")  # This will not be shown due to WARNING level
logger.warning("Everything works but there is something to be aware of.")  # This will be shown
logger.error("There's been a mistake with the process.")  # This will be shown
logger.critical("There is something terribly wrong and process may terminate.")  # This will be shown

logger.remove()
logger.add(
    "analysis/logging/my_log.log", level="DEBUG", rotation="1 MB"
)  # Log all levels to a file and rotate after 1 MB

for i in range(3000):
    logger.debug(f"Debug message {i}")
    logger.info(f"Info message {i}")
    logger.warning(f"Warning message {i}")
    logger.error(f"Error message {i}")
    logger.critical(f"Critical message {i}")
    # This will create multiple log files with rotation after 1 MB


# We can use logger.catch for exception handling
@logger.catch(level="ERROR")
def divide(a, b):
    return a / b


logger.add(sys.stdout, level="ERROR")  # Only show ERROR and above messages on console

divide(10, 0)  # This will log the exception details
