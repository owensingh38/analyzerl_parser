import os
import subprocess
import sys
from pathlib import Path

from .parse import _boxcars_binary, _copy_binary_to_user_cache


def main():
    binary = _boxcars_binary()

    if not sys.platform.startswith("win"):
        mode = os.stat(binary).st_mode
        if mode & 0o111 == 0:
            os.chmod(binary, mode | 0o755)

    command = [binary, *sys.argv[1:]]

    try:
        return subprocess.call(command)
    except PermissionError:
        if sys.platform.startswith("win"):
            raise

        cached_binary = _copy_binary_to_user_cache(Path(binary))
        return subprocess.call([str(cached_binary), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
