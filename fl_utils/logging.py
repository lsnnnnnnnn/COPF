from __future__ import annotations
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

@dataclass
class Logger:
    name: str = "copf"
    level: int = logging.INFO
    log_file: Optional[str] = None  # e.g., "out/run.log"

    def get(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        logger.propagate = False

        if not logger.handlers:
            fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")

            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(self.level)
            ch.setFormatter(fmt)
            logger.addHandler(ch)

            if self.log_file:
                ensure_dir(os.path.dirname(self.log_file) or ".")
                fh = logging.FileHandler(self.log_file)
                fh.setLevel(self.level)
                fh.setFormatter(fmt)
                logger.addHandler(fh)

        return logger