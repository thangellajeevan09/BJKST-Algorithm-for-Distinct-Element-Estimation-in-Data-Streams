"""BJKST distinct-elements estimator, matching the slide deck.

Usage: python bjkst.py cricket_deliveries.csv
"""
import hashlib
import random
import statistics
import sys

import pandas as pd

P = (1 << 61) - 1  # Mersenne prime; hash range is [0, P)


def to_int(item):
    """Stable integer for any item (Python's hash() changes between runs)."""
    return int.from_bytes(hashlib.md5(str(item).encode()).digest()[:8], "big")


class BJKST:
    """One copy. h(x) = ((a*x + b) mod P) / P is pairwise independent."""

    def __init__(self, K, rng):
        self.K = K
        self.a = rng.randrange(1, P)
        self.b = rng.randrange(0, P)
        self.z = 0          # level: keep h(x) < 1/2^z
        self.B = set()      # kept hash values (integers in [0, P))

    def h(self, x):
        return (self.a * to_int(x) + self.b) % P

    def add(self, x):
        v = self.h(x)
        if v * (1 << self.z) < P:            # h(x) < 1/2^z
            self.B.add(v)                    # set: repeats change nothing
            while len(self.B) > self.K:      # overflow: thin out
                self.z += 1
                self.B = {u for u in self.B if u * (1 << self.z) < P}

    def estimate(self):
        return len(self.B) * (1 << self.z)   # |B| * 2^z


def bjkst_median(stream, K, copies, seed=0):
    """Median trick: run independent copies, report the median estimate."""
    rng = random.Random(seed)
    sketches = [BJKST(K, rng) for _ in range(copies)]
    for x in stream:
        for s in sketches:
            s.add(x)
    return statistics.median(s.estimate() for s in sketches)


def toy_example():
    """Slide 7-9 example: fixed 'made up' hashes in sixteenths, K = 3."""
    stream = list("ABACDBEACFDE")
    h16 = {"A": 9, "B": 3, "C": 12, "D": 1, "E": 6, "F": 14}
    K, z, B = 3, 0, set()
    for step, x in enumerate(stream, 1):
        v = h16[x]
        if v * 2 ** z < 16:
            B.add(v)
            while len(B) > K:
                z += 1
                B = {u for u in B if u * 2 ** z < 16}
        print(f"step {step:2d} item {x} h*16={v:2d} z={z} B={sorted(B)}")
    print("Estimate =", len(B) * 2 ** z, "(true = 6)")


if __name__ == "__main__":
    toy_example()
    df = pd.read_csv(sys.argv[1])
    print("\nRows:", len(df))
    for col in ["batter", "bowler", "venue"]:
        stream = df[col].tolist()
        true_d = df[col].nunique()
        print(f"\n{col}: true distinct = {true_d}")
        # c = 144, eps = 0.5 -> K = 576 > d, so the sketch is exact
        print("  K=576 (eps=0.5, c=144):", bjkst_median(stream, 576, 1))
        # small K forces overflow, so the answer is an approximation
        for K in (16, 32, 64):
            est = bjkst_median(stream, K, copies=9, seed=1)
            err = abs(est - true_d) / true_d * 100
            print(f"  K={K:3d}, 9 copies, median est = {est:.0f}  (error {err:.1f}%)")
