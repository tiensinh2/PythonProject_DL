import logging
import logging.handlers
root = logging.getLogger(__name__)
root.setLevel(logging.INFO)
for handler in list(root.handlers):
    root.removeHandler(handler)
format_str = "%(asctime)s %(levelname)-8s pid:%(process)d %(name)s:%(lineno)03d:%(funcName)s %(message)s"
formatter = logging.Formatter(format_str)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)
root.addHandler(stream_handler)
