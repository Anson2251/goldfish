"""Generate the 'repeat' dataset using file-pair format.

File-pair structure (line N of input ↔ line N of output):

  Shard 01 (rangle-langle):  input line = >X,  output line = <X
  Shard 02 (equals):         input line = X=,  output line = X

Each shard pair tests whether the model remembers the content X across
the SEP token boundary.
"""

import string
import random
import os
from typing import Callable

# ── all printable ASCII characters as filler ────────────────────────
chars = (
    string.digits
    + string.ascii_lowercase
    + string.ascii_uppercase
    + "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
)


def _write_shard(d: str, name: str, seed: int, n: int,
                 make_input: Callable, make_output: Callable) -> None:
    """Write input and output files for one shard."""
    rng = random.Random(seed)
    input_lines: list[str] = []
    output_lines: list[str] = []
    for _ in range(n):
        k = rng.randint(1, 6)
        x = "".join(rng.choices(chars, k=k))
        input_lines.append(make_input(x))
        output_lines.append(make_output(x))
    with open(os.path.join(d, name + "-input.txt"), "w") as f:
        f.write("\n".join(input_lines))
    with open(os.path.join(d, name + "-output.txt"), "w") as f:
        f.write("\n".join(output_lines))


def gen_split(d: str, seed: int, n: int) -> None:
    _write_shard(d, "01-rangle-langle", seed, n,
                 make_input=lambda x: f">{x}",
                 make_output=lambda x: f"<{x}")
    _write_shard(d, "02-equals", seed + 10000, n,
                 make_input=lambda x: f"{x}=",
                 make_output=lambda x: x)


# ── generate splits ─────────────────────────────────────────────────
base = os.path.dirname(__file__)

for split, seed, count in [("train", 42, 100_000),
                            ("val", 999, 10_000),
                            ("test", 7777, 10_000)]:
    gen_split(os.path.join(base, split), seed, count)

# report sizes
for split in ("train", "val", "test"):
    for suffix in ("input", "output"):
        for shard in ("01-rangle-langle", "02-equals"):
            path = os.path.join(base, split, f"{shard}-{suffix}.txt")
            size = os.path.getsize(path)
            lines = sum(1 for _ in open(path))
            print(f"{split}/{shard}-{suffix}.txt: {size:>8} bytes, {lines:>8} lines")
