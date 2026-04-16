from __future__ import annotations

import sys
from pathlib import Path


class Tee:
    def __init__(self, file_path: str | Path, stream):
        self.stream = stream
        self.file = open(file_path, "a", encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()