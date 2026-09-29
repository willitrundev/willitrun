#!/usr/bin/env python3
"""Unit tests for willitrun fit math (P0-1 / P0-2). Stdlib only, no GPU, no network.

Run:  python3 scripts/test_fit_math.py
Covers: sharded-GGUF summing (103.7 GiB / 4 shards), KV-cache sizing at 8k/16k/32k,
the MoE two-part offload verdict boundary, and "larger than VRAM+RAM still says no".
"""
import os, sys, tempfile, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("willitrun", os.path.join(HERE, "willitrun.py"))
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)

FAILS = []
def check(name, got, want):
    ok = (got == want) if not isinstance(want, float) else abs(got - want) < 1e-6
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    if not ok: FAILS.append(name)

def approx(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got!r} ~={want!r} (tol {tol})")
    if not ok: FAILS.append(name)

print("1) Sharded-GGUF summing (103.7 GiB split across 4 shards)")
# Real-world split from the operator's Qwen3.8-Flash-Next: 46.44 + 45.99 + 11.26 + 0.01 GiB
parts_gib = [46.44, 45.99, 11.26, 0.01]
want_total = sum(parts_gib) * (1024**3)
with tempfile.TemporaryDirectory() as td:
    names = []
    for i, g in enumerate(parts_gib, start=1):
        nbytes = int(round(g * (1024**3)))
        fn = os.path.join(td, f"Qwen-MoE-0000{i}-of-00004.gguf")
        with open(fn, "wb") as fh:
            fh.seek(nbytes - 1); fh.write(b"\0")   # sparse, no real disk use
        names.append(fn)
    total, shards = W._sum_shards(names[0])
    approx("shard total bytes ~103.70 GiB", total, want_total, int(0.02*(1024**3)))
    check("shard count == 4", shards, 4)

print("2) KV-cache sizing (kv_gib = 2*layers*kvh*hd*ctx*dtype / GiB)")
# 7-9B archetype: layers=32, kv_heads=8, head_dim=128, f16 (2 bytes)
for ctx in (8192, 16384, 32768):
    manual = (2 * 32 * 8 * 128 * ctx * 2) / (1024**3)
    approx(f"kv 7-9B @{ctx//1024}k", W.kv_gib(32, 8, 128, ctx), manual, 1e-9)
# spot value: 8k for that arch is exactly 1.0 GiB
approx("kv 7-9B @8k == 1.0 GiB", W.kv_gib(32, 8, 128, 8192), 1.0, 1e-9)

print("3) MoE two-part verdict boundary (VRAM budget=15.4, avail RAM=35 -> combined=50.4)")
VR, RA = 15.4, 35.0
# exactly fits VRAM -> yes, offload not-needed
check("weights == VRAM budget -> fits_vram", W.offload_verdict(15.4, VR, RA)["fits_vram"], True)
check("weights just over VRAM -> not vram", W.offload_verdict(15.41, VR, RA)["fits_vram"], False)
# within combined -> offload yes
check("weights == combined -> offload yes", W.offload_verdict(50.4, VR, RA)["offload"], "yes")
check("weights just under combined -> offload yes", W.offload_verdict(50.39, VR, RA)["offload"], "yes")

print("4) Larger than VRAM+available RAM still says NO (no over-correction)")
big = W.offload_verdict(103.69, VR, RA)          # the real Qwen size on this box
check("Qwen 103.69 -> offload no", big["offload"], "no")
check("Qwen 103.69 -> not vram fit", big["fits_vram"], False)
# a model bigger than combined by a hair still says no
check("50.41 GiB (>combined) -> offload no", W.offload_verdict(50.41, VR, RA)["offload"], "no")

print("5) Unknown available RAM -> UNKNOWN (never guesses)")
u = W.offload_verdict(103.69, VR, None)
check("ram=None -> offload unknown", u["offload"], "unknown")
check("ram=None -> not vram fit", u["fits_vram"], False)

print("6) Weight math pins the bpw table (a silent constant edit must fail loudly)")
# Q8_0 14B: 14e9 * 8.5 bits / 8 / 1024^3 = 13.85 GiB (matches real on-disk Q8_0 14B GGUFs)
approx("weights_gib(14, Q8_0) == 13.85", W.weights_gib(14.0, "Q8_0"), 13.85, 0.01)
approx("weights_gib(14, Q4_K_M) == 7.90", W.weights_gib(14.0, "Q4_K_M"), 7.90, 0.01)
approx("weights_gib(8, Q6_K) == 6.14", W.weights_gib(8.0, "Q6_K"), 6.14, 0.01)
# unknown quant returns None, never a guess
check("weights_gib(8, Q9_9) is None", W.weights_gib(8.0, "Q9_9") is None, True)

print("7) Hybrid KV: only declared layer types cache (kv_gib_model)")
# Qwen3.8 27B geometry: 64 layers, but config.json declares 16 full_attention and
# 48 linear_attention. Measured against the published config on 2026-09-28.
approx("qwen3.8-27b KV @8k == 0.50 (16 caching layers)",
       W.kv_gib_model(64, 4, 256, 8192, full_layers=16), 0.50, 0.001)
approx("naive overstates it by exactly 4x",
       W.kv_gib(64, 4, 256, 8192) / W.kv_gib_model(64, 4, 256, 8192, full_layers=16), 4.0, 1e-9)
check("full_layers omitted -> naive number (back-compat)",
      W.kv_gib_model(64, 4, 256, 8192) == W.kv_gib(64, 4, 256, 8192), True)

# Gemma 4 12B: 8 full + 40 sliding at window 1024. Sliding layers stop growing.
approx("gemma-4-12b KV @8k == 0.8125 (window honoured)",
       W.kv_gib_model(48, 8, 256, 8192, full_layers=8, sliding_layers=40, window=1024), 0.8125, 0.001)
approx("gemma-4-12b KV @32k == 2.3125",
       W.kv_gib_model(48, 8, 256, 32768, full_layers=8, sliding_layers=40, window=1024), 2.3125, 0.001)
# Beyond the window, growth comes only from the 8 full layers:
# 8 x 2 x kv_heads(8) x head_dim(256) x 2 bytes == 64 KiB per token.
grew = (W.kv_gib_model(48, 8, 256, 32768, full_layers=8, sliding_layers=40, window=1024)
        - W.kv_gib_model(48, 8, 256, 16384, full_layers=8, sliding_layers=40, window=1024))
approx("past the window, +16k tokens costs only the full layers", grew, 1.0, 0.001)
# The naive formula would charge for all 48 layers over that same span.
naive_grew = W.kv_gib(48, 8, 256, 32768) - W.kv_gib(48, 8, 256, 16384)
approx("naive would charge 6x that for the same 16k tokens", naive_grew / grew, 6.0, 1e-9)

# head_dim 0 (GLM 5.3 Flash publishes exactly this) must not become a fake answer.
check("head_dim 0 -> 0.0, and callers treat that as not modelled",
      W.kv_gib_model(45, 64, 0, 8192, full_layers=15, sliding_layers=0) == 0.0, True)

print()
if FAILS:
    print(f"RESULT: {len(FAILS)} FAILED -> {FAILS}"); sys.exit(1)
print("RESULT: all fit-math tests passed")
