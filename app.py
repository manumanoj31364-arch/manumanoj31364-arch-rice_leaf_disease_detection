import os
import sys
import importlib.util

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_SUBDIR = os.path.join(CURRENT_DIR, "Rice Leaf Disease Detection System")

if PROJECT_SUBDIR not in sys.path:
    sys.path.insert(0, PROJECT_SUBDIR)

# Dynamically load the inner app module
app_file = os.path.join(PROJECT_SUBDIR, "app.py")
spec = importlib.util.spec_from_file_location("rice_disease_app", app_file)
inner_module = importlib.util.module_from_spec(spec)
sys.modules["rice_disease_app"] = inner_module
spec.loader.exec_module(inner_module)

# Expose the Flask WSGI application instance
app = inner_module.app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
