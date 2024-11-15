import subprocess
import sys

args = sys.argv[1:]
no_install = args[0] == "-n"
if no_install:
    args = args[1:]
script = (args[0:1] or [""])[0]
if not ".py" in script:
    print("Usage: ./run.sh [-n] <script.py> [args]")
    exit(1)

if not no_install:
    subprocess.call(
        [sys.executable, "-m", "pip", "install", "-r", "requirements_no_torch.txt"]
    )

subprocess.call([sys.executable, "-u", *args])
