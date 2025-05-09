import logging

# Configure once, at import time
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    "%(asctime)s %(name)s %(levelname)s: %(message)s"
))

root = logging.getLogger("MyRLApp")
root.setLevel(logging.DEBUG)
root.addHandler(handler)