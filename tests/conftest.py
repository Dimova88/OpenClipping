import os
import sys
import tempfile

# Isolate test data before any module imports the settings singleton.
_tmp = tempfile.mkdtemp(prefix="openclipping_test_")
os.environ["OPENCLIPPING_DATA"] = _tmp
os.environ["GEMINI_API_KEY"] = ""          # force offline paths in unit tests
os.environ["SCHEDULER_ENABLED"] = "false"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
