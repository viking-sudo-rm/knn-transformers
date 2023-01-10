from typing import Optional
import torch


class PointerLogger:

    """Stops early and logs pointers depending on arguments."""

    def __init__(self, fh: Optional, eval_limit: int, logger=None):
        self.fh = fh
        self.eval_limit = eval_limit
        self.logger = logger
    
    @classmethod
    def open(cls, path: str, eval_limit: int, logger=None):
        if eval_limit == -1:
            return cls(None, eval_limit, logger)
        fh = open(path, "w")
        return cls(fh, eval_limit, logger)

    def update(self, idx, pointers):
        if self.fh is not None:
            # Note: Sorting here removes order: does order matter?
            pointers = sorted(ptr.item() for ptr in pointers)
            line = ",".join(str(ptr) for ptr in pointers) + "\n"
            self.fh.write(line)

        if idx == self.eval_limit:
            if self.fh is not None:
                self.fh.close()
            if self.logger is not None:
                self.logger.info(f"Exiting early after {self.eval_limit} steps.")
            exit()
