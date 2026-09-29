# willitrun

Does this model fit on this machine?

`willitrun` reads your actual hardware — GPU, free VRAM, system RAM, which
inference engines are installed, which model files are already on disk — and tells
you which quantisations of a given model will load, with how much room left. It
prints the command line it would use. It does not install anything.

## Run it

```sh
python3 willitrun.py
```

No arguments needed. Python 3, standard library only — no `pip install`, and
deliberately no `import torch`: importing torch on a GPU that is already full is
one way to crash the machine you are trying to diagnose.

Flags, all optional:

```
--version              print version and exit
--json                 machine-readable output instead of the text report
--save PATH            also write the report to PATH (current directory only)
--no-models            skip the installed-model scan
--claims-db PATH       use a claims.db for tiered citations
--include-hostname     print the real hostname; it is redacted by default
```

## Read it first

The whole point is that you can. One file, no dependencies, nothing obfuscated:

```sh
grep -nE 'urlopen|urllib\.request|requests\.|http\.client|socket\.socket' willitrun.py
```

That returns **nothing** — no network calls of any kind. Worth running yourself
rather than taking this paragraph's word for it.

What the script *does* use, stated plainly because a grep will find them:

- `import socket` (line 34) — one call, `socket.gethostname()` at line 175, for the
  report header. No address, no connection attempt, and the hostname is redacted
  from the printed report unless you pass `--include-hostname`.
- `import subprocess` (line 36) — to run read-only queries like `nvidia-smi`,
  `rocm-smi` and `system_profiler`. Every command is printed before it runs, so the
  transcript doubles as documentation.
- `sqlite3.connect("file:…?mode=ro")` (line 675) — opens LM Studio's model database
  **read-only** to list what you have downloaded.

By default it writes **nothing**. Pass `--save PATH` and it writes the report there,
refusing any path outside the current directory (line 1152). No `sudo`, no PATH
edits, no downloads, no config changes.

## What the numbers mean

A fit verdict is `weights + KV cache for your context length + estimated runtime
overhead`, measured against `total VRAM − 0.5 GiB safety margin`.

Each figure carries a source tier:

| tier | meaning |
| --- | --- |
| `V` | verified — a published config value, or the byte size of the file you would download |
| `C` | computed — parameters × bits-per-weight, where nobody published a file |
| `K` | secondary — from documentation or a vendor page rather than measured here |
| `U` | uncertain — an estimate, such as the runtime overhead term |

Some architectures print **"cache not modelled"** instead of a KV number. That is
deliberate: GLM/DeepSeek MLA and hybrid-Mamba models do not publish a cache shape
the standard formula describes, and inventing one would produce a number that looks
like an answer. Where that happens the total is marked `+kv`, meaning *at least this
much*.

You will not find tokens/sec anywhere in the output. Speed depends on your CPU, your
PCIe lanes, your quantisation kernel and what else is running; a tool that has not
measured your machine has no business quoting it.

## Tests

```sh
python3 test_fit_math.py     # same directory as willitrun.py
```

Covers the arithmetic: bits-per-weight tables, weight sizing, and KV growth for both
uniform-attention and hybrid models, where only a fraction of layers cache at full
context. The same functions are mirrored in JavaScript on
[willitrun.dev](https://willitrun.dev), and the site build refuses to publish unless
all three implementations agree per model.

## Where this comes from

The fit tables, per-GPU pages and per-model pages at
[willitrun.dev](https://willitrun.dev) are generated from the same arithmetic as
this script. The site publishes this file's SHA-256 and byte count on its homepage,
and its build fails if the published digest stops matching the file it serves — so
the copy you download and the copy the numbers were computed from are the same
bytes.

Questions, wrong numbers, missing GPUs: open an issue. A wrong fit number is a bug,
not a caveat, and the source tiers above exist so you can tell me which one it is.
