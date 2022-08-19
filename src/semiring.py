"""Semiring class."""

import abc
import math


class Semiring(metaclass=abc.ABCMeta):

  """Represents an abstract semiring with two operations."""

  def __init__(self, type_, zero, one):
    self.type = type_
    self.zero = zero
    self.one = one

  @abc.abstractmethod
  def add(self, weight1, weight2):
    return NotImplemented

  @abc.abstractmethod
  def mul(self, weight1, weight2):
    return NotImplemented

  @abc.abstractmethod
  def left_divide(self, weight, left):
    return NotImplemented

  @abc.abstractmethod
  def equals(self, weight1, weight2):
    return NotImplemented


class BooleanSemiring(Semiring):

  """The standard boolean semiring for DFAs."""

  def __init__(self):
    super().__init__(bool, False, True)

  def add(self, weight1, weight2):
    return weight1 or weight2

  def mul(self, weight1, weight2):
    return weight1 and weight2

  def left_divide(self, weight, left):
    # TODO(lambdaviking): Could treat divide by 0 here.
    return weight and left

  def equals(self, weight1, weight2):
    return weight1 == weight2


class PlusTimesSemiring(Semiring):

  """The standard plus/times semiring for weighted automata."""

  def __init__(self, eps=0.):
    super().__init__(float, 0., 1.)
    self.eps = eps

  def add(self, weight1, weight2):
    return weight1 + weight2

  def mul(self, weight1, weight2):
    return weight1 * weight2

  def left_divide(self, weight, left):
    return weight / left

  def equals(self, weight1, weight2):
    if self.eps == 0.:
      return weight1 == weight2
    return abs(weight1 - weight2) <= self.eps


class LogSemiring(Semiring):

  """Standard semiring for log probabilities."""

  def __init__(self, eps=0.):
    super().__init__(float, -float("inf"), 0.)
    self.eps = eps

  def add(self, weight1, weight2):
    return math.log(math.exp(weight1) + math.exp(weight2))

  def mul(self, weight1, weight2):
    return weight1 + weight2

  def left_divide(self, weight, left):
    return weight - left

  def equals(self, weight1, weight2):
    p1 = math.exp(weight1)
    p2 = math.exp(weight2)
    if self.eps == 0.:
      return p1 == p2
    return abs(p1 - p2) <= self.eps


class StringSemiring(Semiring):

  """String semiring with concatenation.

  This object technically operates over sets of strings. For efficiency, only
  label final weights with multiple options, not internal weights.

  It is assumed that add is only called on disjoint sets.
  """

  def __init__(self):
    super().__init__(list, [], [""])

  def add(self, weight1: list[str], weight2: list[str]) -> list[str]:
    """Return the union of two finite languages.

    Do NOT call this unless weight1, weight2 are disjoint.

    Also, do not allow non-atomic weights on edges, for efficiency reasons.

    Args:
      weight1: First string to add.
      weight2: Second string to add.

    Returns:
      Union of weight1 and weight2, assuming disjointness.
    """
    return weight1 + weight2

  def mul(self, weight1: list[str], weight2: list[str]) -> list[str]:
    return [w1 + w2 for w1 in weight1 for w2 in weight2]

  def left_divide(self, weight: list[str], left: list[str]) -> list[str]:
    assert len(weight) == 1 and len(left) == 1
    w = weight[0]
    l = left[0]
    assert w.startswith(l)
    return w[len(l):]

  def equals(self, weight1: str, weight2: str) -> bool:
    # Assume sorted and stuff.
    return weight1 == weight2


class PointerSemiring(Semiring):

  # TODO: Bit hacky to get things working.

  def __init__(self):
    super().__init__(list, [], [-1])
  
  def add(self, weight1: list[int], weight2: list[int]):
    return weight1 + weight2

  def mul(self, weight1: list[int], weight2: list[int]):
    # FIXME(lambdaviking): Should this intersect the two lists?
    return NotImplemented

  def equals(self, weight1: list[int], weight2: list[int]) -> bool:
    return weight1 == weight2

  def left_divide(self, weight, left):
    return NotImplemented
