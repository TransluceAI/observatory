import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple


@dataclass
class ColorCode:
    fore: str
    style: str = ""


class Colors:
    # Foreground colors
    BLACK = ColorCode("\033[30m")
    RED = ColorCode("\033[31m")
    GREEN = ColorCode("\033[32m")
    YELLOW = ColorCode("\033[33m")
    BLUE = ColorCode("\033[34m")
    MAGENTA = ColorCode("\033[35m")
    CYAN = ColorCode("\033[36m")
    WHITE = ColorCode("\033[37m")
    BRIGHT_MAGENTA = ColorCode("\033[95m")
    BRIGHT_CYAN = ColorCode("\033[96m")

    # Styles
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @staticmethod
    def apply(text: str, color: ColorCode) -> str:
        return f"{color.style}{color.fore}{text}{Colors.RESET}"


class ColoredFormatter(logging.Formatter):
    COLORS: Dict[int, ColorCode] = {
        logging.DEBUG: Colors.BLUE,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: ColorCode("\033[31m", Colors.BOLD),
    }

    # Available highlight colors
    HIGHLIGHT_COLORS: Dict[str, ColorCode] = {
        "magenta": ColorCode(Colors.BRIGHT_MAGENTA.fore, Colors.BOLD),
        "cyan": ColorCode(Colors.BRIGHT_CYAN.fore, Colors.BOLD),
        "yellow": ColorCode(Colors.YELLOW.fore, Colors.BOLD),
        "red": ColorCode(Colors.RED.fore, Colors.BOLD),
    }

    def __init__(self, fmt: Optional[str] = None) -> None:
        super().__init__(
            fmt or "%(asctime)s [%(levelname)s] %(namespace)s: %(message)s", datefmt="%H:%M:%S"
        )

    def format(self, record: logging.LogRecord) -> str:
        # Add namespace to extra fields if not present
        if not getattr(record, "namespace", None):
            record.__dict__["namespace"] = record.name

        # Color the level name
        record.levelname = Colors.apply(record.levelname, self.COLORS[record.levelno])

        # Color the namespace
        record.__dict__["namespace"] = Colors.apply(record.__dict__["namespace"], Colors.CYAN)

        # Check if highlight flag is set
        highlight = getattr(record, "highlight", None)
        if highlight:
            # Get the highlight color or default to magenta
            color_name = highlight if isinstance(highlight, str) else "magenta"
            highlight_color = self.HIGHLIGHT_COLORS.get(
                color_name, self.HIGHLIGHT_COLORS["magenta"]
            )

            # Apply highlight to the message
            original_message = record.getMessage()
            record.msg = Colors.apply(original_message, highlight_color)
            if record.args:
                record.args = ()

        return super().format(record)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that allows highlighting specific log messages.
    """

    def process(self, msg: Any, kwargs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        # Pass highlight flag through to the record
        return msg, kwargs

    def highlight(
        self,
        msg: object,
        *args: Any,
        color: Literal["magenta", "cyan", "yellow", "red"] = "magenta",
        **kwargs: Any,
    ) -> None:
        """
        Log a highlighted message.

        Args:
            msg: The message format string
            color: The color to highlight with (magenta, cyan, yellow, red)
            *args: The args for the message format string
            **kwargs: Additional logging kwargs
        """
        kwargs.setdefault("extra", {})
        if isinstance(kwargs["extra"], dict):
            kwargs["extra"]["highlight"] = color
        return self.info(msg, *args, **kwargs)


def get_logger(namespace: str) -> LoggerAdapter:
    """
    Get a colored logger for the specified namespace.

    Args:
        namespace: The namespace for the logger

    Returns:
        A configured logger instance with highlighting support
    """
    logger = logging.getLogger(namespace)

    # Only add handler if it doesn't exist
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter())
        logger.addHandler(handler)

        # Set default level to INFO
        logger.setLevel(logging.INFO)

    # Wrap with adapter to support highlighting
    return LoggerAdapter(logger, {})
