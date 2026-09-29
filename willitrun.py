#!/usr/bin/env python3
"""willitrun - read-only local-LLM hardware readiness report.

Reads your GPU/VRAM/RAM/CPU/disk and tells you, in plain English, which model
quantisations actually fit your card with context headroom, what each step up
costs you, and the exact llama.cpp command line for your hardware.

SAFETY CONTRACT (this is a trust product):
  * READ-ONLY. It writes nothing unless you pass --save, and then only one file
    into the current directory. No installs, no sudo, no PATH edits, no downloads.
  * Stdlib + optional nvidia-smi / rocm-smi / system_profiler subprocess calls only.
    It never imports torch (importing torch can itself OOM a loaded GPU).
  * Every external command it runs is printed before it runs.
  * Degrades gracefully: AMD, Apple Silicon, iGPU-only, WSL2, missing drivers all
    still produce a report that states what could not be determined.

Tier tags on numeric claims (the whole brand):
  V = verified off a tagged build's --help / source
  C = official docs, cited and dated
  K = community report (needs a baseline: build + hardware)
  U = unknown; we say so instead of guessing.

Non-affiliation: willitrun is an independent tool. Not affiliated with, endorsed
by, or maintained by Unsloth, ggml.org, Ollama, or any model vendor.
"""

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import socket
import sqlite3
import subprocess
import sys

SCRIPT_VERSION = "2026.09.29-prepush"

QUANT_BPW = {"Q4_K_M": 4.85, "Q6_K": 6.59, "Q8_0": 8.50}   # C - effective bits/weight incl. scales

ARCHETYPES = [
    ("3B",   3.0, 36, 2, 128, "e.g. Qwen2.5-3B class (GQA)"),
    ("7-9B", 8.0, 32, 8, 128, "e.g. Llama-3.1-8B / Qwen2.5-7B class (GQA)"),
    ("14B", 14.0, 48, 8, 128, "e.g. Qwen2.5-14B class (GQA)"),
]

MOE_EXAMPLE = {"label": "35B-A3B MoE", "total_params_b": 35.0, "active_params_b": 3.0}

COMPUTE_BUFFER_GIB = 1.0   # U - typical for mid-size models; higher at big ctx/batch
CUDA_CTX_GIB = 0.4         # K - CUDA/driver context per process (needs baseline)
SAFETY_MARGIN_GIB = 0.5    # our headroom so we never recommend an OOM edge
KV_BYTES_PER_ELEM_F16 = 2  # C - default KV dtype is f16
LLAMA_MOE_FLAGS_TIER = "V"  # llama.cpp MoE/auto-fit flags verified off common/arg.cpp @ master, accessed 2026-09-18:
#   --cpu-moe (all MoE weights in CPU), --n-cpu-moe N (first N layers), --fit [on|off] (auto-adjust args to fit VRAM)


def run(cmd, shell=False):
    """Print then run a read-only command; return stdout string or None. Never raises."""
    argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    disp = " ".join(argv)
    print(f"$ {disp}", file=sys.stderr)  # transparency goes to stderr; stdout stays clean
    try:
        r = subprocess.run(argv, shell=False, capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            err = (r.stderr or "").strip().splitlines()
            if err:
                print(f"   -> exit {r.returncode}: {err[0]}", file=sys.stderr)
            return None
        return r.stdout
    except FileNotFoundError:
        print("   -> command not found", file=sys.stderr)
        return None
    except Exception as e:  # degrade, never crash the diagnostic
        print(f"   -> error: {e}", file=sys.stderr)
        return None


def _to_int(s):
    try:
        return int(re.sub(r"[^\d]", "", str(s)) or "0")
    except Exception:
        return None


def _parse_mib_from_str(s):
    s = (s or "").lower()
    m = re.search(r"(\d+)\s*(gb|gib|mb|mib)", s)
    if not m:
        return _to_int(s)
    n = int(m.group(1)); unit = m.group(2)
    if unit in ("gb", "gib"): return n * 1024
    if unit in ("mb", "mib"): return n
    return None


def detect_gpu():
    sysname = platform.system()
    g = {"backend": None, "vendor": None, "name": None, "total_mib": None,
         "free_mib": None, "used_mib": None, "driver": None, "cuda": None,
         "compute_cap": None, "holders": [], "notes": []}

    if sysname == "Darwin":
        g["backend"] = "apple"; g["vendor"] = "Apple"
        out = run("system_profiler SPDisplaysDataType -json")
        if out:
            try:
                cards = json.loads(out).get("SPDisplaysDataType", [])
                if cards:
                    g["name"] = cards[0].get("sppci_model") or "Apple Silicon GPU"
                    vram = cards[0].get("spdisplays_vram") or cards[0].get("_spdisplays_vram")
                    if vram: g["total_mib"] = _parse_mib_from_str(vram)
            except Exception as e:
                g["notes"].append(f"Could not parse system_profiler output ({e}).")
        g["notes"].append("Apple Silicon uses unified memory: usable VRAM is a fraction of system RAM.")
        return g

    if shutil.which("nvidia-smi"):
        out = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,"
                          "driver_version,compute_cap", "--format=csv,noheader"])
        if out and out.strip():
            parts = [p.strip() for p in out.splitlines()[0].split(",")]
            g["backend"] = "nvidia"; g["vendor"] = "NVIDIA"
            if len(parts) >= 1: g["name"] = parts[0]
            if len(parts) >= 2: g["total_mib"] = _to_int(parts[1])
            if len(parts) >= 3: g["used_mib"] = _to_int(parts[2])
            if len(parts) >= 4: g["free_mib"] = _to_int(parts[3])
            if len(parts) >= 5: g["driver"] = parts[4]
            if len(parts) >= 6: g["compute_cap"] = parts[5]
            ver = run("nvidia-smi --version")
            if ver:
                # Some drivers print "CUDA version : Deprecated"; prefer the UMD line.
                m = re.search(r"CUDA UMD version\s*:\s*([\d.]+)", ver, re.IGNORECASE) \
                    or re.search(r"CUDA version\s*:\s*(\d[\d.]*)", ver, re.IGNORECASE)
                if m: g["cuda"] = m.group(1)
            pout = run(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
                        "--format=csv,noheader"])
            if pout:
                for line in pout.splitlines():
                    seg = [s.strip() for s in line.split(",")]
                    if len(seg) >= 3:
                        g["holders"].append({"pid": seg[0], "mib": _to_int(seg[1]),
                                             "proc": os.path.basename(seg[2])})
            return g
        g["backend"] = "nvidia"
        g["notes"].append("nvidia-smi present but returned no data - driver likely not loaded.")
        return g

    if shutil.which("rocm-smi"):
        out = run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
        g["backend"] = "amd"; g["vendor"] = "AMD"
        if out:
            vals = [v for v in (_to_int(x) for x in re.findall(r"(\d+)\s*$", out, re.MULTILINE)) if v]
            if len(vals) >= 2:
                g["total_mib"] = vals[0] // (1024*1024) if vals[0] > 1_000_000 else vals[0]
                g["used_mib"] = vals[1] // (1024*1024) if vals[1] > 1_000_000 else vals[1]
                if g["total_mib"] and g["used_mib"] is not None:
                    g["free_mib"] = max(0, g["total_mib"] - g["used_mib"])
            g["notes"].append("AMD/ROCm parsing is version-sensitive; treat VRAM figures as approximate (K).")
        if g["total_mib"] is None:
            g["notes"].append("Could not read AMD VRAM reliably. Report proceeds with what was found.")
        return g

    g["backend"] = "none"
    g["notes"].append("No nvidia-smi, rocm-smi, or Apple GPU found. CPU-only inference assumed.")
    return g


def detect_system():
    s = {"os": platform.system(), "release": platform.release(),
         "machine": platform.machine(), "python": platform.python_version(),
         "ram_total_gib": None, "ram_avail_gib": None, "cpu_model": None,
         "cpu_threads": None, "disk_free_gib": None, "wsl": False}
    s["hostname"] = socket.gethostname()
    try:
        with open("/proc/version") as f:
            if "microsoft" in f.read().lower(): s["wsl"] = True
    except Exception:
        pass
    out = run("free -b")
    if out:
        for line in out.splitlines():
            cols = line.split()
            if line.startswith("Mem:") and len(cols) >= 7:
                s["ram_total_gib"] = round(_to_int(cols[1]) / (1024**3), 1)
                s["ram_avail_gib"] = round(_to_int(cols[6]) / (1024**3), 1)
    out = run("lscpu")
    if out:
        for line in out.splitlines():
            if line.lower().startswith("model name"):
                s["cpu_model"] = line.split(":", 1)[1].strip()
            elif re.match(r"^CPU\(s\):", line):
                s["cpu_threads"] = _to_int(line.split(":", 1)[1])
    if not s["cpu_model"]:
        s["cpu_model"] = platform.processor() or "unknown CPU"
    if not s["cpu_threads"]:
        s["cpu_threads"] = os.cpu_count()
    out = run(["df", "-B1", "--output=avail", "/"])
    if out:
        lines = [l for l in out.splitlines() if l.strip().isdigit()]
        if lines: s["disk_free_gib"] = round(int(lines[-1]) / (1024**3), 1)
    return s


def weights_gib(params_b, quant):
    bpw = QUANT_BPW.get(quant)
    if bpw is None or params_b is None: return None
    return params_b * 1e9 * bpw / 8.0 / (1024**3)


def kv_gib(layers, kv_heads, head_dim, ctx):
    """Naive KV cache: every layer stores K and V for every token.

    Correct for plain MHA/GQA. Overstates hybrid-attention models by up to 8x;
    see kv_gib_model(), which is what the scanner should use when a model
    publishes config.json layer_types. Mirrors kvGib() in scripts/lib/fit.mjs.
    """
    return (2 * layers * kv_heads * head_dim * ctx * KV_BYTES_PER_ELEM_F16) / (1024**3)


def kv_gib_model(layers, kv_heads, head_dim, ctx, full_layers=None,
                 sliding_layers=0, window=None):
    """KV cache that honours config.json layer_types.

    full_attention layers grow with context. sliding_attention layers stop
    growing at the window (only true with --flash-attn; without it llama.cpp
    allocates full context for them too). linear_attention and conv layers hold
    a constant recurrent state instead of per-token K/V, so they contribute no
    per-token bytes here -- that state is real memory but its size is not
    published, so it is omitted rather than guessed.

    Pass full_layers=None to get the naive number. Mirrors kvGibModel() in
    scripts/lib/fit.mjs; build-guides.mjs asserts the two agree per model.
    """
    if full_layers is None:
        return kv_gib(layers, kv_heads, head_dim, ctx)
    if not head_dim:
        return kv_gib(layers, kv_heads, head_dim, ctx)
    win = window if window else ctx
    slots = full_layers * ctx + sliding_layers * min(ctx, win)
    return (2 * slots * kv_heads * head_dim * KV_BYTES_PER_ELEM_F16) / (1024**3)


def offload_verdict(weights_gib, vram_budget_gib, ram_avail_gib):
    """Two-part fit verdict as a PURE function (unit-testable).

    Part 1: do the weights alone fit in the VRAM budget?
    Part 2: with expert/CPU offload (some layers/experts on GPU, the rest held in
            system RAM), do they fit in VRAM + AVAILABLE system RAM?
      -> 'yes' | 'no' | 'unknown'. Gated on AVAILABLE RAM, not total; the caller
         must state which basis it used. Never asserts a bare 'too big' when RAM
         could hold the weights.

    Returns {"fits_vram": bool, "offload": 'yes|no|unknown|not-needed', "ram_basis": str|None}.
    """
    fits_vram = (vram_budget_gib is not None and weights_gib <= vram_budget_gib)
    if fits_vram:
        return {"fits_vram": True, "offload": "not-needed", "ram_basis": None}
    if ram_avail_gib is None:
        return {"fits_vram": False, "offload": "unknown", "ram_basis": None}
    combined = (vram_budget_gib or 0.0) + ram_avail_gib
    return {"fits_vram": False,
            "offload": ("yes" if weights_gib <= combined else "no"),
            "ram_basis": f"{ram_avail_gib:g} GiB available RAM"}


def build_fit_table(total_mib):
    total_gib = total_mib / 1024.0 if total_mib else None
    budget = (total_gib - SAFETY_MARGIN_GIB) if total_gib else None
    rows = []
    for label, params_b, layers, kvh, hd, note in ARCHETYPES:
        for quant in ("Q4_K_M", "Q6_K", "Q8_0"):
            w = weights_gib(params_b, quant)
            row = {"model": f"{label} {quant}", "params_b": params_b, "quant": quant,
                   "weights_gib": round(w, 2) if w else None, "arch_note": note, "ctx_fit": {}}
            for ctx in (8192, 16384, 32768):
                kv = kv_gib(layers, kvh, hd, ctx)
                need = (w or 0) + kv + COMPUTE_BUFFER_GIB + CUDA_CTX_GIB
                fits = budget is not None and need <= budget
                headroom = (budget - need) if budget is not None else None
                row["ctx_fit"][ctx] = {"kv_gib": round(kv, 2), "need_gib": round(need, 2),
                                       "fits": fits,
                                       "headroom_gib": round(headroom, 2) if headroom is not None else None}
            rows.append(row)
    return rows, budget


def _recommend_cmd(gpu, sysinfo, budget):
    """Pick the largest archetype+quant that fits at 8k with >=1.5G headroom."""
    best = None
    for label, params_b, layers, kvh, hd, note in ARCHETYPES:
        for quant in ("Q4_K_M", "Q6_K", "Q8_0"):
            w = weights_gib(params_b, quant) or 999
            kv = kv_gib(layers, kvh, hd, 8192)
            need = w + kv + COMPUTE_BUFFER_GIB + CUDA_CTX_GIB
            if budget is not None and need <= budget - 1.5:
                best = (label, params_b, quant, kv)
    if best:
        label, params_b, quant, kv = best
        # choose the largest ctx that still fits with margin
        chosen_ctx = 8192
        for ctx in (32768, 16384, 8192):
            layers, kvh, hd = next((a[2], a[3], a[4]) for a in ARCHETYPES if a[0] == label)
            need = weights_gib(params_b, quant) + kv_gib(layers, kvh, hd, ctx) + COMPUTE_BUFFER_GIB + CUDA_CTX_GIB
            if need <= budget - 1.0:
                chosen_ctx = ctx; break
        cmd = (f"llama-server -m <your-{label}-{quant}.gguf> \\\n"
               f"    --n-gpu-layers 999 \\\n"
               f"    --ctx-size {chosen_ctx} \\\n"
               f"    --flash-attn \\\n"
               f"    --batch-size 512 --ubatch-size 256")
        why = [f"-m: a {label} model at {quant} (largest that fits your card with margin).",
               "--n-gpu-layers 999: offload every layer to GPU; it all fits, so leave none on CPU.",
               f"--ctx-size {chosen_ctx}: largest context that still leaves headroom for KV cache.",
               "--flash-attn: cuts KV-cache memory and speeds long context (build support varies - verify with --help).",
               "--batch/--ubatch: throughput knobs; lower them if you approach an OOM at high ctx."]
        return {"cmd": cmd, "why": why}
    # nothing fits on GPU -> CPU/partial path
    cmd = ("llama-server -m <a-3B-Q4_K_M.gguf> \\\n"
           "    --n-gpu-layers 20 \\\n"
           "    --ctx-size 4096")
    why = ["Your card cannot hold a full model in VRAM; start small and offload only some layers.",
           "--n-gpu-layers: raise gradually until VRAM is nearly full, then back off one layer.",
           "If this still OOMs, use a smaller quant or fewer offloaded layers."]
    return {"cmd": cmd, "why": why}


# ---------------------------------------------------------------------------
# Inference-engine detection + per-engine suggestions.
# All signals are read-only: VRAM-holder process names, the full running-process
# command lines (/proc on Linux, ps on macOS, PowerShell/tasklist on Windows),
# known binaries on PATH via shutil.which, and LISTEN sockets parsed from
# /proc/net/tcp. We map only UNAMBIGUOUS ports to an engine; shared OpenAI-compat
# ports (8000/8080) are never attributed to one engine. Match tokens are launch
# signatures (e.g. "tritonserver", not bare "triton") to avoid false positives.

ENGINES = {
    "llama.cpp":      dict(bins=["llama-server", "llama-cli"], procs=["llama-server", "llama-cli"], mods=[], ports={}),
    "Ollama":         dict(bins=["ollama"], procs=["ollama serve", "ollama run"], mods=["ollama serve"], ports={11434}),
    "LM Studio":      dict(bins=["lms", "lm-studio"], procs=["lm studio", "lmstudio", "lm-studio"], mods=[], ports={1234}),
    "vLLM":           dict(bins=["vllm"], procs=["vllm serve", "vllm.entrypoints", "-m vllm"], mods=["vllm"], ports={}),
    "Jan":            dict(bins=["jan", "cortex"], procs=["jan.exe", "jan.app", "cortex.cpp", "cortex-llamacpp", "cortex.llamacpp", "cortex-cpp"], mods=[], ports={1337}),
    "MLX":            dict(bins=["mlx_lm.server"], procs=[], mods=["mlx_lm", "mlx-lm", "-m mlx_lm"], ports={}),
    "SGLang":         dict(bins=["sglang"], procs=["sglang.launch_server", "-m sglang", "launch_server"], mods=["sglang"], ports={30000}),
    "TensorRT-LLM":   dict(bins=["trtllm-serve", "trtllm-build"], procs=["trtllm"], mods=["tensorrt_llm", "trtllm"], ports={}),
    "ExLlama":        dict(bins=["exllama", "exllamav2-chat", "exllamav3"], procs=["exllama", "tabbyapi"], mods=["exllama", "exllamav2", "exllama_v2", "exllamav3", "tabbyapi", "tabby_api"], ports={}),
    "OpenVINO":       dict(bins=["ovms"], procs=[], mods=["openvino_genai", "optimum-cli openvino", "openvino_model_server", "ovms"], ports={}),
    "Triton":         dict(bins=["tritonserver"], procs=["tritonserver"], mods=[], ports={}),
    "ONNX Runtime":   dict(bins=["onnxruntime_server"], procs=[], mods=["onnxruntime_server", "onnxruntime.serve"], ports={}),
    "Unsloth":        dict(bins=["unsloth", "unsloth-studio"], procs=["unsloth studio", "unsloth-studio", "unsloth_studio"], mods=["unsloth studio", "unsloth.studio"], ports={8888}),
}
# Preferred advice order when multiple are present and none is clearly the one in use.
ENGINE_ORDER = ["llama.cpp", "Unsloth", "Ollama", "LM Studio", "vLLM", "SGLang", "TensorRT-LLM",
                "ExLlama", "Jan", "MLX", "OpenVINO", "Triton", "ONNX Runtime"]

# Advice sourcing (P1-4). Every engine's OFFICIAL doc was fetched on DOC_ACCESS_DATE; the
# 'scope' records exactly what that fetch confirmed. Tuning flags not seen on the fetched
# page are NOT claimed as verified and render as U. See docs/ENGINE_SOURCES.md.
DOC_ACCESS_DATE = "2026-09-18"
ENGINE_SOURCES = {
    "llama.cpp":     ("https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/arg.cpp", "--cpu-moe / --n-cpu-moe N / --fit[on|off] confirmed off source -> V"),
    "Unsloth":       ("https://unsloth.ai/docs/new/studio/install", "'unsloth studio' + install.sh confirmed"),
    "Ollama":        ("https://docs.ollama.com/", "'serve' command confirmed"),
    "LM Studio":     ("https://lmstudio.ai/docs/cli", "'lms' CLI confirmed"),
    "vLLM":          ("", "NOT fetched this session -> U"),
    "SGLang":        ("https://docs.sglang.ai/get_started/install.html", "'launch_server' + port 30000 confirmed"),
    "TensorRT-LLM":  ("https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html", "'trtllm-serve' confirmed"),
    "ExLlama":       ("https://github.com/turboderp-org/exllamav3", "exllamav3 repo confirmed (TabbyAPI server claim = K/U)"),
    "Jan":           ("https://www.jan.ai/docs/desktop/api-server", "local API port 1337 confirmed"),
    "MLX":           ("https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md", "'mlx_lm.server' confirmed"),
    "OpenVINO":      ("https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai.html", "GenAI workflow confirmed (serving subcommand = U)"),
    "Triton":        ("https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html", "'tritonserver' confirmed"),
    "ONNX Runtime":  ("", "NOT fetched this session -> U"),
}
# Engines whose CORE command advice we treat as verified (task-granted + doc fetched):
VERIFIED_ADVICE = {"llama.cpp", "Ollama", "LM Studio"}


def engine_advice_tier(engine):
    """Return the tier for an engine's ADVICE: C if its official doc was fetched this
    session AND it is in VERIFIED_ADVICE; U otherwise (command shape may still be C but
    tuning flags are unverified, surfaced as a caveat)."""
    url = ENGINE_SOURCES.get(engine, ("", ""))[0]
    if engine in VERIFIED_ADVICE and url:
        return "C"
    return "U"


def _pick_fit(budget):
    if budget is None:
        return None
    best = None
    for label, params_b, layers, kvh, hd, note in ARCHETYPES:
        for quant in ("Q4_K_M", "Q6_K", "Q8_0"):
            w = weights_gib(params_b, quant) or 999
            kv = kv_gib(layers, kvh, hd, 8192)
            if w + kv + COMPUTE_BUFFER_GIB + CUDA_CTX_GIB <= budget - 1.5:
                best = (label, params_b, layers, kvh, hd, quant)
    if not best:
        return None
    label, params_b, layers, kvh, hd, quant = best
    chosen = 8192
    for ctx in (32768, 16384, 8192):
        if weights_gib(params_b, quant) + kv_gib(layers, kvh, hd, ctx) + COMPUTE_BUFFER_GIB + CUDA_CTX_GIB <= budget - 1.0:
            chosen = ctx; break
    return {"label": label, "params_b": params_b, "quant": quant, "ctx": chosen}


def _running_cmdlines(lower=True):
    # Best-effort list of running command lines across OSes (lowercased for matching).
    out = []
    sysname = platform.system()
    if sysname == "Linux":
        try:
            for pid in os.listdir("/proc"):
                if not pid.isdigit() or int(pid) == os.getpid():
                    continue
                try:
                    with open(f"/proc/{pid}/cmdline", "rb") as f:
                        raw = f.read().replace(b"\x00", b" ").decode("utf-8", "ignore")
                    if lower:
                        raw = raw.lower()
                    if raw.strip():
                        out.append(raw)
                except Exception:
                    pass
        except Exception:
            pass
    elif sysname == "Darwin":
        r = run("ps -axww -o command=")
        if r:
            out += [(l.lower() if lower else l) for l in r.splitlines()]
    elif sysname == "Windows":
        r = run('powershell -NoProfile -Command "Get-CimInstance Win32_Process | ForEach-Object { $_.CommandLine }"')
        if r:
            out += [(l.lower() if lower else l) for l in r.splitlines()]
        else:
            r = run("tasklist /fo csv /nh")   # image names only (best-effort fallback)
            if r:
                out += [(l.lower() if lower else l) for l in r.splitlines()]
    return out


def _listening_ports():
    ports = set()
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(f) as fh:
                next(fh, None)
                for line in fh:
                    p = line.split()
                    if len(p) >= 4 and p[3] == "0A":
                        try:
                            ports.add(int(p[1].rsplit(":", 1)[-1], 16))
                        except ValueError:
                            pass
        except Exception:
            pass
    return ports


def detect_engines(gpu):
    in_use, seen = [], set()

    def mark(eng, evidence):
        if eng not in seen:
            in_use.append({"engine": eng, "evidence": evidence}); seen.add(eng)

    # 1) VRAM-holder process names (strongest: loaded on the GPU right now).
    for h in gpu.get("holders", []):
        pn = (h.get("proc") or "").lower()
        for eng, spec in ENGINES.items():
            if any(tok in pn for tok in spec["procs"] + spec["bins"]):
                mark(eng, f"process '{h['proc']}' holding VRAM")

    # 2) full running-process command lines (catches engines not holding VRAM here).
    cmdlines = _running_cmdlines()
    joined = "\n".join(cmdlines)
    for eng, spec in ENGINES.items():
        if eng in seen:
            continue
        tokens = spec["procs"] + spec["mods"]
        if any(tok in joined for tok in tokens):
            mark(eng, "running process")

    # 3) unambiguous listening ports.
    ports = _listening_ports()
    for eng, spec in ENGINES.items():
        if eng in seen:
            continue
        hit = [p for p in spec["ports"] if p in ports]
        if hit:
            mark(eng, f"listening on :{hit[0]}")

    # 4) installed binaries on PATH (present but not necessarily running).
    installed = []
    for eng, spec in ENGINES.items():
        found = [b for b in spec["bins"] if shutil.which(b)]
        if found:
            installed.append({"engine": eng, "binaries": found})

    primary = None
    if in_use:
        # prefer the engine that holds VRAM, else first by stable order
        holder_engines = [e["engine"] for e in in_use if "holding VRAM" in e["evidence"]]
        pool = holder_engines or [e["engine"] for e in in_use]
        primary = sorted(pool, key=lambda e: ENGINE_ORDER.index(e))[0]
    elif installed:
        pool = [i["engine"] for i in installed]
        primary = sorted(pool, key=lambda e: ENGINE_ORDER.index(e))[0]
    return {"in_use": in_use, "installed": installed, "primary": primary}


def _suggest_engine(engine, gpu, sysinfo, budget):
    if engine == "llama.cpp" or engine is None:
        return _recommend_cmd(gpu, sysinfo, budget)

    fit = _pick_fit(budget)
    if not fit:
        return {"kind": "cli",
                "cmd": f"# {engine}: your card cannot hold a full model in VRAM.",
                "why": ["Start with a 3B Q4_K_M and the smallest context that works.",
                        "If it still OOMs, drop the quant or run CPU-only."]}

    label, quant, ctx = fit["label"], fit["quant"], fit["ctx"]

    if engine == "Ollama":
        qtag = quant.lower()
        cmd = f"ollama pull <model>:{qtag}\nollama run <model>:{qtag}"
        why = [f"Model: a {label} at {quant} (largest that fits your card with margin).",
               "Set context once via a Modelfile so it survives restarts - write these lines:",
               "    FROM ./<model>.gguf",
               f"    PARAMETER num_ctx {ctx}",
               "    PARAMETER num_gpu 999",
               "Then: ollama create <name> -f Modelfile && ollama run <name>",
               "Env knobs (C): OLLAMA_FLASH_ATTENTION=1, OLLAMA_KV_CACHE_TYPE=q8_0 (halves KV bytes),",
               "OLLAMA_NUM_PARALLEL=1 on a single-GPU box so two requests do not fight for VRAM.",
               "If a request still OOMs, lower num_ctx before lowering num_gpu."]
        return {"kind": "cli", "cmd": cmd, "why": why}

    if engine == "vLLM":
        cmd = (f"vllm serve <model> \\\n"
               f"    --max-model-len {ctx} \\\n"
               f"    --gpu-memory-utilization 0.90")
        why = [f"Model: a {label} at {quant}. For GGUF weights add --quantization gguf_v2.",
               "--max-model-len sets the context; it also sizes vLLM's KV pool, so this is your main lever.",
               "--gpu-memory-utilization 0.90 leaves room for activations + CUDA context on one card.",
               "CAVEAT (U): vLLM PREALLOCATES a KV pool up front, so the headroom math above (built",
               "   for llama.cpp) is only approximate here - trust vLLM's own startup log for real free VRAM.",
               "Single 16 GB slot: keep --tensor-parallel-size 1; do not fan out."]
        return {"kind": "cli", "cmd": cmd, "why": why}

    if engine == "SGLang":
        cmd = (f"python -m sglang.launch_server \\\n"
               f"    --model-path <model> \\\n"
               f"    --mem-fraction-static 0.90 \\\n"
               f"    --context-length {ctx}")
        why = [f"Model: a {label} (SGLang serves HF weights; for GGUF prefer llama.cpp).",
               "--context-length is the main VRAM lever; --mem-fraction-static caps the KV pool share.",
               "CAVEAT (U): like vLLM, SGLang preallocates a KV pool - trust its startup log for real free VRAM.",
               "Single GPU: leave default TP=1; do not add --tp-size on one card."]
        return {"kind": "cli", "cmd": cmd, "why": why}

    if engine == "TensorRT-LLM":
        cmd = f"trtllm-serve <hf-model> --max_batch_size 1"
        why = [f"Model: a {label}. TensorRT-LLM usually needs an offline engine build (trtllm-build)",
               "   sized to YOUR VRAM before serving - that compile step is outside this read-only tool.",
               "--max_batch_size 1 on a single 16 GB card; larger batches need more activation memory.",
               "CAVEAT (U): built engines are fixed-size and lock GPU memory at load; headroom math differs.",
               "This is the highest-effort engine here - only worth it if you need its throughput."]
        return {"kind": "cli", "cmd": cmd, "why": why}

    if engine == "Unsloth":
        cmd = (f"unsloth studio -p 8888\n"
               f"# open http://127.0.0.1:8888 and load a {label} {quant} model")
        why = ["Unsloth Studio/Desktop is a free GUI over llama.cpp (and MLX on Apple); it downloads",
               "   GGUF/MLX models and runs them locally on Mac, Windows and Linux.",
               f"Pick a {label} at {quant}; set GPU offload to the max that fits, back off one layer if it OOMs.",
               "OpenAI-compatible API via Settings -> API; default binds 127.0.0.1:8888.",
               "Install (if missing): curl -fsSL https://unsloth.ai/install.sh | sh   then run: unsloth studio",
               "(V) launch/port per official docs unsloth.ai/docs/new/studio. Not affiliated with Unsloth."]
        return {"kind": "gui", "cmd": cmd, "why": why}

    if engine == "ExLlama":
        cmd = f"# ExLlamaV3 (v2 archived): serve via TabbyAPI -> python tabbyapi.py --host 127.0.0.1 --port 5000"
        why = ["ExLlama uses .exl2/.exl3 files, which are NOT the GGUF quants in the fit table above.",
               "The recommended server is TabbyAPI (FastAPI); set max_context_len and per-GPU memory split there.",
               "Quantization is baked into the exl2/exl3 file - pick a lower bpw file to fit, not a runtime flag.",
               "(K) ExLlamaV3 is current; ExLlamaV2 is archived. Strong on NVIDIA chat throughput; Apple/AMD limited."]
        return {"kind": "cli", "cmd": cmd, "why": why}

    if engine == "Jan":
        cmd = f"Jan (GUI): download a {label} model, then set parameters in Settings."
        why = ["Jan runs llama.cpp under the hood with a GUI; its local OpenAI API is on :1337.",
               f"- Model Parameters: set GPU layers to max that fits this quant; back off one if it OOMs.",
               f"- Context size: about {ctx} tokens.",
               "- Download the Q4_K_M/Q6_K build of the model Jan offers for your card.",
               "If a request OOMs, lower context before lowering GPU layers."]
        return {"kind": "gui", "cmd": cmd, "why": why}

    if engine == "MLX":
        cmd = (f"mlx_lm.server --model <repo-or-path> \\\n"
               f"    --kv-bits 8 --kv-group-size 64")
        why = ["MLX is Apple Silicon only (unified memory): the VRAM-fit table above is approximate here",
               "   because MLX shares system RAM with the GPU. Treat free SYSTEM RAM as your budget (U).",
               f"Model: a {label} quant; --kv-bits 8 halves KV-cache bytes for long context.",
               "Use mlx_lm.generate / mlx_lm.chat for one-off runs, mlx_lm.server for an OpenAI-compat API.",
               "On Apple silicon prefer 4-bit (mlx-community) builds sized to your RAM headroom."]
        return {"kind": "cli", "cmd": cmd, "why": why}

    if engine == "OpenVINO":
        cmd = (f"optimum-cli export openvino --model <repo> --weight-format int4 <out-dir>\n"
               f"# then serve: ovms --model_name m --model_path <out-dir>   (or an OpenVINO GenAI pipeline)")
        why = ["OpenVINO targets Intel CPU / iGPU / NPU, not CUDA - the VRAM-fit table does NOT apply;",
               "   budget against system RAM and your accelerator instead (U).",
               "Export HF weights to OpenVINO IR (.xml/.bin) with optimum-cli; int4 keeps them small.",
               "Serve via OVMS or an OpenVINO GenAI pipeline; recent optimum-intel also adds an 'openvino serve' subcommand (K).",
               "Good fit for Intel boxes without discrete VRAM; not the path for an RTX card."]
        return {"kind": "cli", "cmd": cmd, "why": why}

    if engine == "Triton":
        cmd = f"tritonserver --model-repository <dir> --model-load-thread-count 4"
        why = ["NVIDIA Triton is a serving framework: you export/convert models into a repository layout",
               "   first; it does not run a bare GGUF/HF path directly.",
               "Size instances per model so total GPU memory stays under your card (16 GB here); start with 1 instance.",
               "CAVEAT (U): Triton preallocates per-instance memory - trust its logs for real free VRAM.",
               "Setup-heavy; usually overkill for a single-user local box."]
        return {"kind": "cli", "cmd": cmd, "why": why}

    if engine == "ONNX Runtime":
        cmd = f"# ONNX Runtime is a library: run .onnx models via onnxruntime-gpu or behind Triton/FastAPI."
        why = ["No single server command - ORT is embedded in an app or served via Triton.",
               "Use the CUDAExecutionProvider on NVIDIA; set enable_cuda_graph + memory arena to fit VRAM.",
               "The model scan below lists any .onnx files present. Quantized ONNX (int8) fits smaller cards.",
               "Note (U): exact VRAM depends on your graph and provider settings, not a fixed formula."]
        return {"kind": "cli", "cmd": cmd, "why": why}

    base = _recommend_cmd(gpu, sysinfo, budget)
    base["why"] = [f"(No specific guidance for '{engine}' yet; showing the llama.cpp equivalent.)"] + base["why"]
    return base


def _do_not_try(gpu, sysinfo, budget):
    out = []
    if gpu["backend"] == "none":
        out.append("Expecting GPU speed on CPU-only: large models will be seconds-per-token, not tokens-per-second.")
        return out
    total_gib = (gpu.get("total_mib") or 0) / 1024.0
    if budget is None:
        out.append("Trusting these numbers blindly - VRAM could not be read; verify with your own tools.")
        return out
    w14_q8 = weights_gib(14.0, "Q8_0")
    if w14_q8 and (w14_q8 + COMPUTE_BUFFER_GIB) > budget:
        out.append(f"14B at Q8_0 (~{w14_q8:.1f} GiB weights): does not fit; it will offload or OOM.")
    w9_q4 = weights_gib(8.0, "Q4_K_M")
    if w9_q4 and (w9_q4 + kv_gib(32, 8, 128, 32768) + COMPUTE_BUFFER_GIB) > budget:
        out.append("A 7-9B model at full 32k context on top of everything else: KV cache alone pushes it over.")
    w_moe = weights_gib(MOE_EXAMPLE["total_params_b"], "Q4_K_M")
    if w_moe and (w_moe + COMPUTE_BUFFER_GIB) > budget:
        out.append(f"Large MoE models ({MOE_EXAMPLE['label']}, ~{w_moe:.1f} GiB weights): active-params is small but weight VRAM is not.")
    if total_gib < 12:
        out.append("Running two models at once or a server plus heavy desktop use; you are too close to the ceiling.")
    if not out:
        out.append("Nothing obvious - your card has room. The paid check would look for driver/flag mismatches, which this read-only scan cannot see.")
    return out


def _claims_citations(path):
    """Optional: pull a couple of tiered flag citations from book-project claims.db."""
    if not path or not os.path.isfile(path):
        return []
    lines = []
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        cur = con.cursor()
        for flag in ("--ctx-size", "--n-gpu-layers"):
            row = cur.execute(
                "SELECT status, strength, substr(text_snippet,1,80) FROM claim "
                "WHERE canonical_flag=? ORDER BY (status='valid') DESC LIMIT 1", (flag,)).fetchone()
            if row:
                tier = {"valid": "C/V", "needs-baseline": "K"}.get(row[0], "U")
                snippet = re.sub(r"[`\r\n]+", " ", row[2] or "")
                snippet = re.sub(r"\s+", " ", snippet).strip()
                if len(snippet) > 72:
                    snippet = snippet[:72].rsplit(" ", 1)[0] + "..."
                lines.append(f"corpus {flag} [{tier}/{row[1]}]: {snippet}")
        con.close()
    except Exception as e:
        lines.append(f"(claims.db present but unreadable: {e})")
    return lines


# ---------------------------------------------------------------------------
# Installed-model discovery (read-only, bounded, cross-platform).
# Uses MEASURED on-disk sizes (tier V), never guesses params. Skips tokenizer /
# vocab files and the /media air-gap path; caps time + depth so it stays fast.

_MODEL_QUANT_BPW = {  # C - effective bits/weight, for naming only (sizes are measured)
    "Q2_K":2.6,"Q3_K_S":3.4,"Q3_K_M":3.9,"Q4_0":4.5,"Q4_K_S":4.5,"Q4_K_M":4.85,
    "Q5_K_S":5.7,"Q5_K_M":5.7,"Q6_K":6.59,"Q8_0":8.5,"IQ4_XS":4.25,"F16":16.0,"BF16":16.0,
}


def _model_roots():
    home = os.path.expanduser("~")
    env = os.environ
    roots = [f"{home}/models", f"{home}/weights", f"{home}/llama.cpp/models",
             f"{home}/.ollama/models", f"{home}/.cache/llama.cpp",
             f"{home}/.cache/huggingface", f"{home}/.cache/lm-studio",
             f"{home}/LM-Studio/models", f"{home}/Machine-Learning-Models", f"{home}/Jan/models", "/models"]
    for e in ("HF_HOME", "OLLAMA_MODELS"):
        if env.get(e):
            roots.append(env[e])
    # operator-supplied extra model dirs (pathsep-separated) for non-standard layouts
    for extra in env.get("WILLITRUN_MODEL_DIRS", "").split(os.pathsep):
        if extra.strip():
            roots.append(extra.strip())
    sysname = platform.system()
    if sysname == "Darwin":
        roots += [f"{home}/Library/Application Support/LM Studio/models", f"{home}/.lmstudio/models"]
    elif sysname == "Windows":
        up = env.get("USERPROFILE", home); la = env.get("LOCALAPPDATA", "")
        roots += [f"{up}\\.ollama\\models", f"{up}\\.cache\\huggingface\\hub"]
        if la:
            roots.append(f"{la}\\Programs\\LM Studio\\models")
    out, seen = [], set()
    for r in roots:
        try:
            rp = os.path.abspath(os.path.expanduser(r))
        except Exception:
            continue
        if not rp or rp in seen or rp.startswith("/media/"):   # air-gap guard
            continue
        seen.add(rp)
        if os.path.isdir(rp):
            out.append(rp)
    return out


def _infer_quant(name):
    m = re.search(r"(IQ\d_[A-Z]+|Q\d_[A-Z]?_?[KMSL]|\bQ\d_\d\b|Q\d_K_M|BF16|F16)", name, re.IGNORECASE)
    if not m:
        return None
    tok = m.group(0).upper()
    return tok


def _models_from_processes():
    # Ground truth: model paths named on running engine command lines (-m/--model/...).
    # Preserves case, so call with lower=False.
    found = set()
    for cl in _running_cmdlines(lower=False):
        toks = re.split(r"\s+", cl.strip())
        for i, t in enumerate(toks):
            if t in ("-m", "--model", "--model-path", "--model_path", "--gptq", "--adapter") and i + 1 < len(toks):
                p = toks[i + 1].strip('"\'')
                if p:
                    found.add(p)
    return found


def _sum_shards(gguf_path):
    # Sum all shard files of a sharded GGUF that share the de-sharded stem.
    d = os.path.dirname(gguf_path) or "."
    stem = re.sub(r"-0*\d+-of-0*\d+", "", os.path.basename(gguf_path))
    total, shards = 0, 0
    try:
        for fn in os.listdir(d):
            if fn.lower().endswith(".gguf") and re.sub(r"-0*\d+-of-0*\d+", "", fn) == stem:
                try:
                    total += os.path.getsize(os.path.join(d, fn)); shards += 1
                except Exception:
                    pass
    except Exception:
        pass
    if total == 0:
        try:
            total = os.path.getsize(gguf_path)
        except Exception:
            return None, 0
    return total, max(1, shards)


def find_models(budget, max_seconds=None, top=20):
    import time as _t
    if max_seconds is None:
        try:
            max_seconds = float(os.environ.get("WILLITRUN_MODEL_SCAN_SECONDS", "12"))
        except ValueError:
            max_seconds = 12.0
    deadline = _t.time() + max_seconds
    roots = _model_roots()
    agg = {}                 # key -> {size, shards, path, kind, quant}
    model_files_seen = 0     # individual model files matched (shards counted separately)
    walked, completed = [], []
    partial, stopped_in = False, None

    def add(key, size, full, kind, quant=None):
        cur = agg.get(key)
        if cur is None:
            agg[key] = {"size": size, "shards": 1, "path": full, "kind": kind, "quant": quant}
        else:
            cur["size"] += size; cur["shards"] += 1

    # A) running-process model paths (always attempted; not subject to the walk budget).
    for p in _models_from_processes():
        try:
            rp = os.path.abspath(os.path.expanduser(p))
        except Exception:
            continue
        if rp.startswith("/media/"):
            continue
        lowp = rp.lower()
        if lowp.endswith(".gguf") and "vocab" not in lowp:
            size, shards = _sum_shards(rp)
            if size:
                model_files_seen += max(1, shards)
                key = "G:" + re.sub(r"-0*\d+-of-0*\d+", "", os.path.basename(rp))
                add(key, size, rp, "gguf", _infer_quant(os.path.basename(rp)))
        elif lowp.endswith(".safetensors"):
            d = os.path.dirname(rp)
            tot = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.lower().endswith(".safetensors")) if os.path.isdir(d) else 0
            model_files_seen += 1
            add("S:" + d, tot or (os.path.getsize(rp) if os.path.exists(rp) else 0), rp, "safetensors")

    # B) bounded directory scan of known model stores. Walk roots and RECORD what we
    #    actually entered vs where a time limit stopped us - never claim a root we did not read.
    scanned = 0
    for root in roots:
        walked.append(root)
        base = root.rstrip("/\\"); pre = len(base); finished_root = True
        for dp, dns, fns in os.walk(base, followlinks=False):
            if _t.time() > deadline:
                partial = True; stopped_in = root; finished_root = False
                break
            if dp[pre:].count("/") > 12 and dp[pre:].count("\\") > 12:
                dns[:] = []; continue
            dns[:] = [d for d in dns if not any(s in d.lower() for s in (".git","node_modules",".venv","site-packages","blobs"))]
            for fn in fns:
                scanned += 1
                low = fn.lower()
                full = os.path.join(dp, fn)
                if low.endswith(".gguf") and "vocab" not in low and "tokenizer" not in low:
                    try: size = os.path.getsize(full)
                    except Exception: continue
                    model_files_seen += 1
                    key = "G:" + re.sub(r"-0*\d+-of-0*\d+", "", fn)
                    if key not in agg:
                        add(key, size, full, "gguf", _infer_quant(fn))
                elif low.endswith(".onnx"):
                    try: size = os.path.getsize(full)
                    except Exception: continue
                    model_files_seen += 1
                    add("O:" + full, size, full, "onnx")
        if finished_root:
            completed.append(root)
        else:
            break   # time is up; do not pretend later roots were read

    models = []
    for key, info in agg.items():
        gib = info["size"] / (1024**3)
        name = os.path.basename(info["path"].rstrip("/\\"))
        if info["kind"] == "gguf":
            name = re.sub(r"-0*\d+-of-0*\d+", "", name)
            if info["shards"] > 1:
                name += f" ({info['shards']} shards)"
            eng = "llama.cpp / Ollama / LM Studio / Jan"
        elif info["kind"] == "safetensors":
            name = os.path.basename(os.path.dirname(info["path"].rstrip("/\\"))) or name
            eng = "vLLM / SGLang / TensorRT-LLM / MLX (or convert for llama.cpp)"
        else:
            eng = "ONNX Runtime / Triton"
        models.append({"name": name, "path": info["path"].replace(os.path.expanduser("~"), "~"),
                       "size_gib": round(gib, 2), "kind": info["kind"], "quant": info.get("quant") or "U",
                       "engines": eng})

    models.sort(key=lambda m: -m["size_gib"])
    return {"roots_walked": [r.replace(os.path.expanduser("~"), "~") for r in walked],
            "roots_completed": [r.replace(os.path.expanduser("~"), "~") for r in completed],
            "partial": partial,
            "stopped_in": (stopped_in.replace(os.path.expanduser("~"), "~") if stopped_in else None),
            "model_files_seen": model_files_seen, "files_seen": scanned,
            "models": models[:top], "truncated": len(models) > top}


def render(gpu, sysinfo, fit_rows, budget, citations, engines, models=None):
    L = []
    bar = "=" * 68
    L += [bar, " willitrun - local-LLM hardware readiness (read-only)", bar]
    L.append(f" Host:     {sysinfo.get('hostname')}")
    os_line = f" OS:       {sysinfo['os']} {sysinfo['release']} ({sysinfo['machine']})"
    if sysinfo["wsl"]: os_line += "  [WSL2 detected]"
    L.append(os_line)
    L.append(f" Python:   {sysinfo['python']}")
    L.append("")
    L.append("-- Your hardware -------------------------------------------")
    if gpu["backend"] == "nvidia":
        L.append(f" GPU:      {gpu['name']}  (CUDA compute capability {gpu.get('compute_cap') or 'unknown'})")
        L.append(f" Driver:   {gpu.get('driver') or 'unknown'}   CUDA runtime reported: {gpu.get('cuda') or 'unknown'}")
    elif gpu["backend"] == "amd":
        L.append(f" GPU:      {gpu['name'] or 'AMD GPU (model not parsed)'}  [ROCm]")
    elif gpu["backend"] == "apple":
        L.append(f" GPU:      {gpu['name'] or 'Apple Silicon'}  [unified memory]")
    else:
        L.append(" GPU:      none detected - CPU-only inference")
    if gpu["total_mib"]:
        L.append(f" VRAM:     {gpu['total_mib']} MiB total | {gpu.get('used_mib')} MiB used | {gpu.get('free_mib')} MiB free right now")
    else:
        L.append(" VRAM:     could not determine (see notes below)")
    if gpu["holders"]:
        L.append(" Currently holding VRAM:")
        for h in gpu["holders"]:
            L.append(f"   - PID {h['pid']}: {h['proc']} using ~{h['mib']} MiB")
    L.append(f" RAM:      {sysinfo['ram_total_gib']} GiB total | {sysinfo['ram_avail_gib']} GiB available")
    L.append(f" CPU:      {sysinfo['cpu_model']} ({sysinfo['cpu_threads']} threads)")
    if sysinfo["disk_free_gib"] is not None:
        L.append(f" Disk:     {sysinfo['disk_free_gib']} GiB free on /")
    for n in gpu["notes"]:
        L.append(f" note:     {n}")
    L.append("")

    if gpu["backend"] == "nvidia" and gpu.get("free_mib") is not None and gpu["total_mib"]:
        if (gpu["free_mib"] / gpu["total_mib"]) < 0.15:
            L.append("** Heads up: your card is almost fully occupied right now.")
            L.append(f"   Only {gpu['free_mib']} MiB of {gpu['total_mib']} MiB is free.")
            L.append("   The fit table below is judged against the card's idle capacity,")
            L.append("   but you must UNLOAD whatever holds VRAM before loading a new model:")
            for h in gpu["holders"]:
                L.append(f"     kill {h['pid']}  # or stop it from your server UI")
            L.append("")

    L.append("-- Detected inference engine(s) ----------------------------")
    if engines["in_use"]:
        for e in engines["in_use"]:
            L.append(f" In use:     {e['engine']}  ({e['evidence']})")
    else:
        L.append(" In use:     none detected running on this GPU right now.")
    inst = [i["engine"] for i in engines["installed"]]
    if inst:
        L.append(f" Installed:  {', '.join(inst)} (found on PATH)")
    else:
        L.append(" Installed:  none detected on PATH")
    if engines["primary"]:
        L.append(f" Advice targets: {engines['primary']}")
    else:
        L.append(" No engine detected. Easiest install for your card:")
        L.append("   Unsloth Studio/Desktop (free GUI, GGUF + MLX, Mac/Windows/Linux):")
        L.append("     curl -fsSL https://unsloth.ai/install.sh | sh    then run:  unsloth studio")
        L.append("   Alternatives: Ollama (https://ollama.com), or llama.cpp built from source.")
        L.append(" A llama.cpp default is shown below until an engine is present. (V) install cmd per unsloth.ai/docs")
    L.append("")

    L.append("-- What fits (against idle card capacity) -------------------")
    if budget is None:
        L.append(" Cannot compute fit: VRAM total unknown. See notes.")
    else:
        L.append(f" Budget: {budget:.1f} GiB usable (= total minus {SAFETY_MARGIN_GIB} GiB safety margin).")
        L.append(" KV-cache dtype assumed f16 (C). Overheads are estimates (U/K).")
        L.append(" NOTE: this table is judged against VRAM ONLY. Configs that offload experts/layers to")
        L.append("       system RAM are evaluated separately in the installed-model scan below.")
        L.append("")
        header = " " + f"{'Model':<14}{'Weights':>8}  " + "".join(f"{str(c//1024)+'k ctx':>14}" for c in (8192, 16384, 32768))
        L.append(header)
        L.append(" " + "-" * (len(header) - 1))
        for r in fit_rows:
            cells = ""
            for ctx in (8192, 16384, 32768):
                cf = r["ctx_fit"][ctx]
                mark = f"FIT +{cf['headroom_gib']:.1f}G" if cf["fits"] else "no"
                cells += f"{mark:>14}"
            wtxt = (str(r['weights_gib']) + 'G') if r['weights_gib'] else '?'
            L.append(f" {r['model']:<14}{wtxt:>8}  {cells}")
        L.append("")
        moe = MOE_EXAMPLE
        wq4 = weights_gib(moe["total_params_b"], "Q4_K_M")
        fits_moe = (wq4 + COMPUTE_BUFFER_GIB + CUDA_CTX_GIB) <= budget
        L.append(f" MoE note: a {moe['label']} stores ~{wq4:.1f} GiB of weights at Q4_K_M even")
        L.append(f"   though only ~{moe['active_params_b']}B is active per token. Weight VRAM scales")
        L.append(f"   with TOTAL params, not active. {'It fits.' if fits_moe else 'It does NOT fit in weight VRAM alone.'}")
    L.append("")

    if models is not None:
        ram_avail = sysinfo.get("ram_avail_gib")
        L.append("-- Installed models found (read-only scan) -----------------")
        walked = models.get("roots_walked", [])
        if models.get("partial"):
            stop = models.get("stopped_in") or "an unknown root"
            L.append(f" Scanned (PARTIAL SCAN - time limit reached, stopped in {stop}):")
            L.append(f"   roots read: {', '.join(walked) if walked else 'none'} (+ running engine model paths).")
            L.append("   Incomplete. Widen it: WILLITRUN_MODEL_DIRS=/path/to/models  (colon/; separated)")
            L.append("   or raise the budget: WILLITRUN_MODEL_SCAN_SECONDS=30 (default 12).")
        elif walked:
            L.append(f" Scanned: {', '.join(walked)} (+ running engine model paths).")
        if not models.get("models"):
            L.append(" No model files found in the scanned locations (+ running engine paths).")
            L.append(' Set WILLITRUN_MODEL_DIRS for non-standard model directories.')
        else:
            L.append(f" Found {len(models['models'])} model(s) from {models.get('model_files_seen', 0)} file(s)."
                     + (" (list truncated to the largest)" if models.get("truncated") else ""))
            for mm in models["models"]:
                q = f"  q={mm['quant']}" if mm.get("quant") and mm["quant"] != "U" else ""
                L.append(f"  {mm['size_gib']:>7} GiB [{mm['kind']}{q}]  {mm['name'][:48]}")
                v = offload_verdict(mm["size_gib"], budget, ram_avail)
                if v["fits_vram"]:
                    hr = (budget - mm["size_gib"]) if budget is not None else None
                    hrs = f" (+{hr:.1f} GiB headroom)" if hr is not None else ""
                    L.append(f"           -> fits in VRAM alone: yes{hrs}; runs on: {mm['engines']}")
                    continue
                else:
                    basis = v["ram_basis"] or "system RAM unknown"
                    combined = (budget or 0) + (ram_avail or 0)
                    more = mm["size_gib"] - combined
                    L.append(f"           -> fits in VRAM alone: no")
                    if v["offload"] == "yes":
                        L.append(f"              expert/CPU offload into {basis} (+VRAM): YES - weights fit resident")
                    elif v["offload"] == "no":
                        L.append(f"              expert/CPU offload into {basis} (+VRAM): NO - ~{more:.0f} GiB short resident")
                        L.append(f"              mmap/expert streaming may still run it from disk via page cache;")
                        L.append(f"              feasibility + speed UNMEASURED (U)")
                    else:
                        L.append(f"              expert/CPU offload: UNKNOWN - available RAM not reported")
                    # Source: llama.cpp common/arg.cpp @ master, accessed 2026-09-18.
                    # When the per-layer CPU split can't be inferred, keep ALL MoE weights in CPU RAM
                    # (--cpu-moe) and let --fit on auto-adjust the remaining args to VRAM.
                    L.append(f"              mechanism [{LLAMA_MOE_FLAGS_TIER}]: llama.cpp --cpu-moe --fit on")
                    L.append("                (keep all MoE weights in CPU RAM; --fit on auto-fits the rest to VRAM)")
                    L.append("                if the per-layer split is known, use --n-cpu-moe N instead.")
                    L.append("              speed penalty for offload: unmeasured (U) - benchmark is the paid follow-up")
        L.append("")
        L.append(" Sizes are MEASURED on disk (V). VRAM-only fit ignores CPU-offloaded configs, which are")
        L.append(" evaluated separately above against AVAILABLE system RAM. HF/safetensors weights must be")
        L.append(" converted for llama.cpp, or served via vLLM / SGLang / TensorRT-LLM / MLX.")
        L.append("")

    L.append("-- What each step up costs ---------------------------------")
    if gpu["backend"] == "nvidia":
        L += [" Going Q4_K_M -> Q6_K -> Q8_0 raises weight bytes ~35% then ~29%.",
              " Inference at typical batch is memory-bandwidth-bound (C): tokens/sec",
              " scale roughly with bytes read per token, so each step trades speed for",
              " quality. Exact tok/s on YOUR card requires a measured benchmark - that",
              " is the paid follow-up; we do not print a number we have not measured (U).",
              "",
              " Quality you actually buy per step:",
              "   Q4_K_M: best size/quality tradeoff; occasional bumps on math/reasoning",
              "           and rare-token reproduction.",
              "   Q6_K:   near-Q8 quality, ~35% more VRAM than Q4 for small gains.",
              "   Q8_0:   ~lossless vs fp16; usually the wrong choice when it costs you",
              "           context length or forces CPU offload (net slower)."]
    else:
        L += [" Directional only: higher quants cost proportionally more memory and, once",
              " you spill to CPU/RAM, far more time. Exact figures need a measured",
              " benchmark on your hardware (U)."]
    L.append("")

    primary = engines.get("primary") or "llama.cpp"
    ptier = engine_advice_tier(primary)
    disp = engines.get("primary") or "llama.cpp default"
    L.append(f"-- Suggested starting point ({disp}) --------------- [advice tier: {ptier}]")
    sug = _suggest_engine(engines.get("primary"), gpu, sysinfo, budget)
    L.append(sug["cmd"])
    L.append("")
    for why in sug["why"]:
        L.append("   " + why)
    if ptier == "U":
        src_url = ENGINE_SOURCES.get(primary, ("", ""))[0] or "no doc fetched this session"
        L.append(f"   NOTE (U): command shape from official docs ({src_url}, accessed {DOC_ACCESS_DATE});")
        L.append(f"         specific tuning flags for {primary} were NOT individually verified this run.")
    L.append("")
    others, seenp = [], {primary}
    for e in engines.get("in_use", []):
        if e["engine"] not in seenp:
            others.append(e["engine"]); seenp.add(e["engine"])
    for i in engines.get("installed", []):
        if i["engine"] not in seenp:
            others.append(i["engine"]); seenp.add(i["engine"])
    if others:
        L.append("-- Also on this machine (one-line config) ------------------")
        for eng in others:
            s = _suggest_engine(eng, gpu, sysinfo, budget)
            first = s["cmd"].splitlines()[0].rstrip().rstrip("\\").rstrip()
            tag = "" if engine_advice_tier(eng) == "C" else "  [advice unverified - U]"
            L.append(f" {eng}: {first}{tag}")
        L.append("")

    L.append("-- Do NOT try this on your hardware ------------------------")
    for d in _do_not_try(gpu, sysinfo, budget):
        L.append("   - " + d)
    L.append("")

    if citations:
        L.append("-- Evidence from the reference corpus (tiered) -------------")
        for c in citations:
            L.append("   " + c)
        L.append("")

    L.append("-- Method & honesty ----------------------------------------")
    L += [" Weights = params x bits/weight (C). KV = 2 x layers x kv_heads x head_dim",
          " x ctx x dtype (C). Overheads are estimates and marked U/K; where we do not",
          " know, we say unknown rather than guess. No GPU affiliate links, no cloud",
          " upsell - when we say your card cannot do something well, there is no angle.",
          "",
          " willitrun is an independent tool. Not affiliated with, endorsed by, or",
          " maintained by Unsloth, ggml.org, Ollama, or any model vendor."]
    L.append(bar)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(
        prog="willitrun", description="Read-only local-LLM hardware readiness report.")
    ap.add_argument("--save", metavar="PATH", help="also write the report to PATH (current dir only).")
    ap.add_argument("--claims-db", metavar="PATH", help="optional path to a claims.db for tiered citations.")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of the text report.")
    ap.add_argument("--no-models", action="store_true", help="skip the installed-model scan (section B).")
    ap.add_argument("--include-hostname", action="store_true",
                    help="print the real hostname (it is redacted by default for privacy).")
    ap.add_argument("--version", action="version", version="%(prog)s " + SCRIPT_VERSION,
                    help="print the script version and exit.")
    args = ap.parse_args()

    print("willitrun read-only probes: nvidia-smi / rocm-smi / system_profiler; free;", file=sys.stderr)
    print("lscpu; df; uname. Engine detection also reads running-process command lines", file=sys.stderr)
    print("(/proc/<pid>/cmdline on Linux, ps/Get-CimInstance on macOS/Windows) and LISTEN", file=sys.stderr)
    print("sockets (/proc/net/tcp). It makes NO network calls, installs nothing, and writes", file=sys.stderr)
    print("nothing unless you pass --save (current dir only). Hostname is redacted by default.", file=sys.stderr)

    gpu = detect_gpu()
    sysinfo = detect_system()
    if not args.include_hostname:
        sysinfo["hostname"] = "(redacted - use --include-hostname)"
    fit_rows, budget = build_fit_table(gpu.get("total_mib"))
    engines = detect_engines(gpu)
    models = find_models(budget) if not args.no_models else None
    citations = _claims_citations(args.claims_db)

    if args.json:
        payload = {"gpu": gpu, "system": sysinfo,
                   "budget_gib": round(budget, 2) if budget else None,
                   "engines": engines, "models": models,
                   "fit": fit_rows}
        text = json.dumps(payload, indent=2, default=str)
    else:
        text = render(gpu, sysinfo, fit_rows, budget, citations, engines, models)

    print("\n" + text)

    if args.save:
        path = os.path.abspath(args.save)
        if os.path.dirname(path) != os.getcwd():
            print(f"\n[refusing to write outside the current directory: {args.save}]")
            sys.exit(2)
        else:
            try:
                with open(path, "w") as f:
                    f.write(text + "\n")
                print(f"\n[saved report to {path}]")
            except Exception as e:
                print(f"\n[could not save report: {e}]")
                sys.exit(2)

    # exit codes: 0 = report produced; 1 = no GPU detected (report still printed)
    if not gpu.get("name"):
        sys.exit(1)


if __name__ == "__main__":
    main()
