from typing import Optional
import torch


class PointerLogger:

    """Stops early and logs pointers depending on arguments."""

    def __init__(self, fh : Optional, eval_limit: int):
        self.fh = fh
        self.eval_limit = eval_limit
    
    @classmethod
    def open(cls, path: str, eval_limit: int):
        if eval_limit == -1:
            return cls(None, eval_limit)
        fh = open(path, "w")
        return cls(fh, eval_limit)

    def log(self, pointers) -> None:
        """Log pointers and return whether to break."""
        if self.fh is not None:
            # Note: Sorting here removes order: does order matter?
            pointers = sorted(ptr.item() for ptr in pointers)
            line = ",".join(str(ptr) for ptr in pointers) + "\n"
            self.fh.write(line)

    def done(self, idx: int) -> bool:
        if idx == self.eval_limit:
            if self.fh is not None:
                self.fh.close()
            return True
        return False
