import logging
import os
import sys
from colorama import init, Fore, Style      # tiny cross-platform helper

init(strip=False)                           # initialise once; no-op on TTY-less streams

# --- 1.  Colour palette -----------------------------------------------------
_COLOURS = [
    Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE,
    Fore.MAGENTA, Fore.CYAN, Fore.WHITE
]


def _pick_colour(pid: int) -> str:
    """Map a PID to one of the palette colours deterministically."""
    return _COLOURS[pid % len(_COLOURS)]


# --- 2.  Custom formatter ---------------------------------------------------
class ProcessColourFormatter(logging.Formatter):
    """Wrap each log line in a colour chosen per-process."""
    def format(self, record: logging.LogRecord) -> str:
        # Keep the original formatted text
        base = super().format(record)

        # If stderr isn't a TTY (e.g. redirected to file) – skip colours
        if not sys.stderr.isatty():
            return base

        colour = _pick_colour(record.process)
        return f"{colour}{base}{Style.RESET_ALL}"


root = logging.getLogger("MyRLApp")
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

fmt = "%(asctime)s  %(name)s  %(levelname)s: %(message)s"
handler.setFormatter(ProcessColourFormatter(fmt))

root.addHandler(handler)
root.propagate = False

# # Configure once, at import time
# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)
# handler.setFormatter(logging.Formatter(
#     "%(asctime)s %(name)s %(levelname)s: %(message)s"
# ))
#
# root = logging.getLogger("MyRLApp")
# root.setLevel(logging.DEBUG)
# root.addHandler(handler)