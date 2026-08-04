from __future__ import annotations

from dataclasses import dataclass


MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
STATE_SIZE = 624


@dataclass
class OfficialMt19937:
    words: list[int]
    index: int = STATE_SIZE
    draw_count: int = 0

    def copy(self) -> "OfficialMt19937":
        return OfficialMt19937(self.words.copy(), self.index, self.draw_count)


def fold_seed32(seed: int) -> int:
    value = (seed + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    value ^= value >> 31
    folded = ((value >> 32) ^ value) & MASK32
    return folded or 1


def seed_mt19937(seed: int) -> OfficialMt19937:
    words = [0] * STATE_SIZE
    words[0] = seed & MASK32
    for index in range(1, STATE_SIZE):
        previous = words[index - 1]
        words[index] = (
            1812433253 * (previous ^ (previous >> 30)) + index
        ) & MASK32
    return OfficialMt19937(words)


def twist_mt19937(engine: OfficialMt19937) -> None:
    n = STATE_SIZE
    m = 397
    upper_mask = 0x80000000
    lower_mask = 0x7FFFFFFF
    a = 0x9908B0DF
    for k in range(n - m):
        value = (engine.words[k] & upper_mask) | (engine.words[k + 1] & lower_mask)
        engine.words[k] = engine.words[k + m] ^ (value >> 1) ^ (a if value & 1 else 0)
    for k in range(n - m, n - 1):
        value = (engine.words[k] & upper_mask) | (engine.words[k + 1] & lower_mask)
        engine.words[k] = engine.words[k + m - n] ^ (value >> 1) ^ (a if value & 1 else 0)
    value = (engine.words[n - 1] & upper_mask) | (engine.words[0] & lower_mask)
    engine.words[n - 1] = engine.words[m - 1] ^ (value >> 1) ^ (a if value & 1 else 0)
    engine.index = 0


def mt19937_next(engine: OfficialMt19937) -> int:
    if engine.index >= STATE_SIZE:
        twist_mt19937(engine)
    value = engine.words[engine.index]
    engine.index += 1
    engine.draw_count += 1
    value ^= value >> 11
    value ^= (value << 7) & 0x9D2C5680
    value ^= (value << 15) & 0xEFC60000
    value ^= value >> 18
    return value & MASK32


def uniform_below(engine: OfficialMt19937, range_: int) -> int:
    if not 0 < range_ <= MASK32:
        raise ValueError("range must be in [1, 2**32 - 1]")
    product = mt19937_next(engine) * range_
    low = product & MASK32
    if low < range_:
        threshold = ((-range_) & MASK32) % range_
        while low < threshold:
            product = mt19937_next(engine) * range_
            low = product & MASK32
    return (product >> 32) & MASK32
