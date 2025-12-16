import sys

# Absolute path to the existing project we are reusing.
EXTERNAL_APP_ROOT = "/mnt/c/Users/wes/Desktop/target_credit_b/src"

if EXTERNAL_APP_ROOT not in sys.path:
    sys.path.append(EXTERNAL_APP_ROOT)
