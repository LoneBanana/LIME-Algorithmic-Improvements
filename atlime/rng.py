"""Deterministic pseudo-random numbers: Mulberry32 + Box-Muller.

Implemented from scratch instead of using the standard-library ``random``
module so that the exact same stream can be reproduced bit-for-bit by the
TypeScript port that powers the website (``web/lib/core/rng.ts``).
"""

from __future__ import annotations

import math

_MASK = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    """32-bit integer multiply (the semantics of JavaScript's ``Math.imul``)."""
    return (a * b) & _MASK


class Rng:
    """Mulberry32 generator with a cached Box-Muller Gaussian."""

    __slots__ = ("_state", "_spare")

    def __init__(self, seed: int = 0) -> None:
        self._state = seed & _MASK
        self._spare: float | None = None

    def next_u32(self) -> int:
        self._state = (self._state + 0x6D2B79F5) & _MASK
        a = self._state
        t = _imul(a ^ (a >> 15), a | 1)
        t = ((t + _imul(t ^ (t >> 7), t | 61)) & _MASK) ^ t
        return (t ^ (t >> 14)) & _MASK

    def uniform(self) -> float:
        """Uniform float in [0, 1)."""
        return self.next_u32() / 4294967296.0

    def normal(self) -> float:
        """Standard normal via the Box-Muller transform (second value cached)."""
        if self._spare is not None:
            z = self._spare
            self._spare = None
            return z
        u1 = 1.0 - self.uniform()  # (0, 1], keeps log finite
        u2 = self.uniform()
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        self._spare = r * math.sin(theta)
        return r * math.cos(theta)

    def randint(self, n: int) -> int:
        """Uniform integer in [0, n)."""
        return int(self.uniform() * n)

    def shuffle(self, seq: list) -> None:
        """In-place Fisher-Yates shuffle."""
        for i in range(len(seq) - 1, 0, -1):
            j = self.randint(i + 1)
            seq[i], seq[j] = seq[j], seq[i]

    def spawn(self, salt: int) -> "Rng":
        """Independent child generator; deterministic in (parent draw, salt)."""
        return Rng(derive_seed(self.next_u32(), salt))


def derive_seed(seed: int, salt: int) -> int:
    """Mix two integers into a new 32-bit seed (one round of Mulberry32)."""
    return Rng((seed * 0x9E3779B1 + salt * 0x85EBCA77) & _MASK).next_u32()
