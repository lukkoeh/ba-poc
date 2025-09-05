import re, os, hashlib, json, time, orjson
from typing import Optional

def extract_id(filename: str) -> Optional[str]:
    # Expects ..._<ID>.<ext>, where ID is last underscore segment before extension
    base = os.path.basename(filename)
    m = re.search(r"_([A-Za-z0-9]+)\.[^.]+$", base)
    return m.group(1) if m else None

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def now_ts() -> int:
    return int(time.time())

def dumps_json(data) -> str:
    return orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()

def get_git_hash() -> Optional[str]:
    try:
        import subprocess
        return subprocess.check_output(["git","rev-parse","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

