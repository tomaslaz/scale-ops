---
layout: distill
title: "First Principles vs. Reality: Dense vs. MoE Qwen3.6 on a DGX Spark"
description: "A roofline model built purely from published hardware specs and model architecture predicts real vLLM throughput on a DGX Spark to within ~5%, explaining why MoE's real advantage over dense is 4.2x (not the naive 9x) and why neither model can profitably serve inference on a single box."
permalink: /dgx_spark_dense_vs_moe/
category: "Benchmarking and Performance"
date: 2026-08-18
future: true
htmlwidgets: true
hidden: false

giscus_comments: false

authors:
  - name: Tomas

toc:
  - name: The hardware DGX Spark (GB10)
    anchor: the-hardware-dgx-spark-gb10
    subsections:
      - name: References
        anchor: references
  - name: The two models
    anchor: the-two-models
  - name: The setup
    anchor: the-setup
  - name: Memory footprint
    anchor: memory-footprint
  - name: Reverse-engineering the KV cost
    anchor: reverse-engineering-the-kv-cost
  - name: The measured results
    anchor: the-measured-results
  - name: The roofline model
    anchor: the-roofline-model
    subsections:
      - name: Does it predict what was measured?
        anchor: does-it-predict-what-was-measured
  - name: "The expert-overlap puzzle — solved two ways"
    anchor: the-expert-overlap-puzzle--solved-two-ways
  - name: Can you actually make money doing this?
    anchor: can-you-actually-make-money-doing-this
    subsections:
      - name: "At the throughput actually measured (batch=4)"
        anchor: at-the-throughput-actually-measured-batch4
      - name: Concurrency, realistically
        anchor: concurrency-realistically
---

NVIDIA's DGX Spark is marketed as an AI supercomputer for your desk, and it's had plenty of reviews and benchmarks since launch — most of them fairly positive. Assuming the hardware holds up, though, a box like this sitting on my desk is going to be idle a lot of the time. So I am wondering: would it be reasonable, or even profitable, to sell inference off a DGX Spark? It's a £5,000 machine. If it's idle most of the time, that's not a great return on investment.

Answering it turned into something more interesting: a chance to check whether a back-of-envelope roofline model can actually predict what a real vLLM server does, down to the token.

This post walks through both: the empirical benchmark, and the first-principles math that explains it.

> **Key takeaways**
>
> - A roofline model built purely from published hardware specs and model architecture predicts real vLLM throughput to within **~5%**, using only one fitted constant (effective memory bandwidth).
> - That fitted bandwidth comes out to **~230 GB/s — about 84% of the 273 GB/s nominal spec** — a real-world efficiency gap in the same ballpark as the compute side, where measured BF16/FP4 also land well below rated TFLOPS.
> - MoE's real measured advantage over dense is **4.2×**, not the naive **9×** you'd expect from "3B active vs. 27B active" — the gap is fully explained by expert overlap once you account for batching.
> - vLLM appears to be sizing KV cache as if every layer needs a growing attention cache, DeltaNet layers included — a **~4× memory-cost gap** that, if fixed, would meaningfully raise both models' usable context.
> - **No, you can't profitably sell inference off a single DGX Spark.** Dense loses money under any realistic scenario; MoE only clears its costs in an unrealistic best case (sustained near-100% utilization at high concurrency) and falls back underwater the moment demand looks like real-world traffic.

## The hardware: DGX Spark (GB10)

| Spec                 | Value                                                          |
| -------------------- | -------------------------------------------------------------- |
| GPU Architecture     | Blackwell, SM_121 [1]                                          |
| CUDA Cores           | 6,144 [1]                                                      |
| CPU                  | 20-core Arm (10× Cortex-X925 + 10× Cortex-A725) [1]            |
| FP4 (Tensor, sparse) | 1,000 TOPS rated · ~480 TFLOPS measured [2][3]                 |
| BF16 (Tensor)        | ~125 TFLOPS rated · ~60 TFLOPS measured [2][3]                 |
| Memory               | 128 GB unified LPDDR5x, shared CPU+GPU [1]                     |
| Memory Bandwidth     | 273 GB/s nominal [1] · ~230 GB/s effective (fitted, see below) |
| Power                | 140 W SoC TDP / 240 W system rated [2]                         |
| Price                | £4,959.98 (Aug 2026, without discounts/promotions) [4]         |

For context, a single Blackwell GPU in NVIDIA's datacenter-class [GB200 Superchip](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) has roughly 20–42× Spark's compute and ~29–35× its memory bandwidth (2,500 TFLOPS BF16, 8 TB/s HBM3e, per official specs) — but the compute:bandwidth _ratio_ that actually determines memory- vs. compute-bound behavior in a roofline model is surprisingly similar: ~312 FLOP/byte for GB200 vs. ~261 FLOP/byte for Spark using the measured figures above. The memory-bound story this whole post is built around isn't a quirk of small desktop machine — the same dynamic holds at full datacenter scale too.

### References

1. DGX Spark User Guide — ["Hardware Overview"](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
2. TechPowerUp — ["NVIDIA DGX Spark Reportedly Runs at Half the Power and Performance"](https://www.techpowerup.com/342321/nvidia-dgx-spark-reportedly-runs-at-half-the-power-and-performance) — rated 1,000 TOPS FP4 / 125 TFLOPS BF16; measured ~480 TFLOPS FP4 / ~60 TFLOPS BF16; 140W SoC / 240W system power.
3. John Carmack on X (@ID_AA_Carmack), Oct 27, 2025 — [original post](https://x.com/ID_AA_Carmack/status/1982831774850748825)
4. https://www.scan.co.uk/

## The two models

Both are from Alibaba's Qwen3.6 family, both hybrid architectures mixing **Gated DeltaNet** (linear attention, fixed-size recurrent state) with **standard gated attention** (the kind with a real, growing KV cache) — but structured very differently.

|                                      | Qwen3.6-27B (dense)                                         | Qwen3.6-35B-A3B (MoE)                                       |
| ------------------------------------ | ----------------------------------------------------------- | ----------------------------------------------------------- |
| Parameters                           | 27B                                                         | 35B total, 3B active                                        |
| Total layers                         | 64                                                          | 40                                                          |
| Layer layout                         | 16 × (3 DeltaNet : 1 Attention) → 48 DeltaNet, 16 attention | 10 × (3 DeltaNet : 1 Attention) → 30 DeltaNet, 10 attention |
| Gated Attention (KV heads, head_dim) | 4 KV heads, head dim 256                                    | 2 KV heads, head dim 256                                    |
| Experts                              | —                                                           | 256 routed + 1 shared, top-8 routed active/token            |
| Context (native / YaRN)              | 262,144 / 1,010,000                                         | 262,144 / 1,010,000                                         |

_YaRN extends usable context beyond the natively trained length by interpolating RoPE frequencies._

## The setup

- **Host:** spark-a902 (GB10, ~122 GiB usable unified memory)
- **Stack:** vLLM (`vllm/vllm-openai:cu130-nightly`), BF16, `max_model_len 262144`, `gpu_memory_utilization 0.9`, prefix caching on
- **Benchmark:** `vllm bench serve`, 32 requests × (4,096 input / 512 output tokens), max concurrency 4

## Memory footprint

Using the setup above, the measured memory footprint for each model is as reported in the vLLM startup logs:

|                   | 27B (dense)    | 35B-A3B (MoE)  |
| ----------------- | -------------- | -------------- |
| Model weights     | 51.1 GiB       | 65.5 GiB       |
| KV cache pool     | 49.9 GiB       | 37.8 GiB       |
| KV cost per token | ~257 KiB       | ~80 KiB        |
| KV cache capacity | 203,840 tokens | 494,208 tokens |

Both weight sizes check out exactly against BF16 (2 bytes/param): 65.5 GiB ÷ 2 ≈ 35.2B, 51.1 GiB ÷ 2 ≈ 27.4B. **The KV numbers are where it gets interesting.**

## Reverse-engineering the KV cost

The per-token KV formula:

```
bytes/token = n_attention_layers × 2 (K,V) × n_kv_heads × head_dim × bytes/elem
```

Each new token's attention layers need to look back at every previous token's Key and Value vectors, so K and V are computed once per token and cached rather than recomputed from scratch. `n_kv_heads × head_dim` is the size of one K (or V) vector — both models use grouped-query attention, where several query heads share fewer KV heads, so the cache scales with the smaller KV-head count, not the query-head count. The `× 2` accounts for storing K and V separately, `bytes/elem` is the numeric precision (2 bytes for BF16), and it's `n_attention_layers` rather than total layers because DeltaNet layers keep a fixed-size recurrent state instead of a growing cache — only true attention layers contribute here.

Plugging in each model's true attention-layer count and KV heads from the table above, counting attention layers only:

```
Dense: 16 × 2 × 4 × 256 × 2 bytes = 64 KiB/token
MoE:   10 × 2 × 2 × 256 × 2 bytes = 20 KiB/token
```

Both land at **25% of the measured cost** from the memory footprint above (257 KiB dense, 80 KiB MoE) — the same 4× gap on two models with entirely different layer counts and head configs, which rules out coincidence.

Both models interleave 3 DeltaNet layers per 1 attention layer, so `total_layers ÷ attention_layers = 4` for each (64÷16, 40÷10) — suspiciously the same number as the gap.

**Working hypothesis:** vLLM is sizing KV cache as if every layer needs a real, growing attention cache — including DeltaNet layers, which carry only a fixed-size recurrent state and shouldn't need one at all. That would be a gap in vLLM's hybrid-model handling, not a property of the models themselves. It's inferred from matching the numbers, not confirmed against vLLM's source, so treat it as a leading hypothesis rather than settled fact.

**Why it matters:** the earlier claim that "dense can't even cache its own 262K context" (204K < 262K) assumes 257 KiB/token is the true attention cost. If DeltaNet layers are being costed unnecessarily, dense's real capacity is roughly 4× higher — past the 262K native context, though still short of the 1.01M YaRN-extended length.

**Bottom line on capacity:** as currently measured, MoE holds **~2.4× the context length per GB of KV cache** that dense does (494,208 vs. 203,840 tokens) — despite having a _smaller_ KV pool (37.8 GiB vs. 49.9 GiB), since more of its memory budget goes to weights. That 2.4× gap is why MoE comfortably fits its own 262,144-token native context (1.9× over) while dense falls short of its own (0.78×) — it can't serve its full advertised context window under this setup. If the every-layer-costed hypothesis above turns out to be correct and gets fixed, both figures scale up ~4× (dense to ~815K, MoE to ~1.98M) and the 2.4× ratio between them holds — but dense would then also clear its native context, and MoE's corrected capacity would exceed even the 1.01M YaRN-extended length.

## The measured results

Running the setup above through `vllm bench serve`:

|                                   | 27B (dense)      | 35B-A3B (MoE)  | MoE advantage |
| --------------------------------- | ---------------- | -------------- | ------------- |
| Output throughput (aggregate)     | 15.7 tok/s       | 65.9 tok/s     | 4.2×          |
| Per-stream speed (1/TPOT)         | 4.1 tok/s        | 18.0 tok/s     | 4.4×          |
| Mean TTFT                         | 6.8 s            | 2.7 s          | 2.5×          |
| P99 TTFT                          | 14.8 s           | 6.9 s          | 2.1×          |
| Peak KV cache usage (estimated)\* | 3.2% (~6.5K tok) | 1.6% (~8K tok) | —             |

_TTFT = time to first token (prefill latency); TPOT = time per output token (decode latency)._

**\*Estimated, not directly logged:** vLLM doesn't print a peak-usage percentage in this output; these figures are back-calculated from observed peak KV memory use divided by the per-token KV cost established above. They describe the same measurement the KV-cost figures already capture, not an independent one — so they're useful context for how little of the pool this benchmark actually touched, but shouldn't be read as a second, separate confirmation of the KV-cost numbers.

**The surprise:** a naive read of "3B active vs. 27B active" predicts a **9×** MoE advantage. The measured throughput advantage is only **4.2×** — something is eating over half of MoE's theoretical edge. Notably, the TTFT advantage (2.1–2.5×) is _smaller_ than the decode-throughput advantage (4.2–4.4×) — consistent with MoE's prefill (a single 4,096-token prompt routing through top-8 of 256 experts, token by token) very likely touching most of its experts anyway, eroding the "3B active" edge specifically during prompt processing.

## The roofline model

A roofline model is a simple performance-estimation technique borrowed from HPC: rather than simulating everything a system does, it predicts execution time as bounded by whichever hardware limit — raw compute throughput or memory bandwidth — is the tighter constraint for the workload in question.

It's relevant here because it lets us predict vLLM's serving throughput for both models directly from published hardware specs and model architecture. The version below is borrowed from [Reiner Pope's first-principles breakdown of inference economics](https://gist.github.com/dwarkeshsp/79100f0fdeed69d76241903bb0604dbe):

```
t = max(t_compute, t_mem)
t_compute = 2 × B × N_active / FLOPS
t_mem     = t_weights + t_KV
t_weights = N_total × bytes_per_param / BW
t_KV      = B × L × bytes_per_token / BW
```

`t_compute` is how long the step would take if memory were infinitely fast — pure arithmetic time. `t_mem` is the opposite: how long it'd take to just _stream_ everything the step needs — every weight, plus each sequence's KV cache — if the matrix multiplies themselves were free. A GPU overlaps compute with memory transfer, so the real step time is gated by whichever is slower, not their sum — hence `max`, not `+`.

For single-token decode at small batch sizes, `t_mem` almost always wins: there's very little arithmetic per token, but the model's full weights still have to move through memory on every step. That's the bet a memory-bound roofline model is making, and it's what gets tested below.

Every term is a published or measured constant — `N_active`/`N_total` (parameter counts), `bytes_per_param` (2 for BF16), `B` (batch = 4), `L` (average context length over the run), `bytes_per_token` (the measured KV cost from the section above) — except one: `BW`, the memory bandwidth actually achieved in practice, which is never quite the datasheet number.

**Fitting `BW`.** Dense is the cleanest place to solve for it: it's unambiguously memory-bound, with no MoE expert-overlap complication (that puzzle comes later). Rearranging the formula gives `BW = (weights bytes + KV bytes) / t`, where `t` comes straight from dense's measured throughput: `t = B / 15.7 tok/s ≈ 255 ms`. Plugging in dense's published 27B params and its measured 257 KiB/token KV cost:

```
BW ≈ (54e9 + 4.6e9) bytes / 0.255 s ≈ 230 GB/s
```

— about 84% of the 273 GB/s nominal spec. That's the one free parameter in the whole model; every other prediction below, including MoE's, reuses this same fitted 230 GB/s plus published model constants — no further tuning.

### Does it predict what was measured?

|                                   | Dense 27B      | MoE 35B-A3B    |
| --------------------------------- | -------------- | -------------- |
| `t_compute`                       | 3.6 ms         | 0.4 ms         |
| `t_weights`                       | 235 ms         | — (see below)  |
| `t_KV`                            | 20 ms          | 6 ms           |
| Predicted `t = t_mem`             | **255 ms**     | **61 ms**\*    |
| Predicted aggregate tok/s (`B/t`) | 15.7 tok/s     | 65.9 tok/s\*   |
| **Measured aggregate tok/s**      | **15.7 tok/s** | **65.9 tok/s** |
| Match                             | **~100%**      | **~100%**\*    |

\*MoE's weight-fetch time isn't the full 35B — see below.

**Dense matches essentially exactly, as expected — it's the calibration point.** MoE's match looks equally good, but only after fitting its **effective active parameter count**, since the naive "3B active" assumption doesn't survive contact with batch>1.

## The expert-overlap puzzle — solved two ways

At batch=4, how much weight MoE actually streams per step depends on overlap between the four requests' expert choices: fully disjoint routing costs 4×3B=12B, full overlap costs just 3B. Reality sits in between — and pinning down where explains the gap between the naive 9× MoE advantage and the measured 4.2×.

**Method 1 — back-solve from the benchmark.** Using the fitted 230 GB/s bandwidth and the observed 65.9 tok/s, the implied effective active weight at batch=4 is **~6.3B**.

**Method 2 — predict it from the routing config alone**, no benchmark involved. With 256 experts and top-8 routing, the expected number of distinct experts touched across 4 requests is:

```
E[distinct] = 256 × (1 − (1 − 8/256)⁴) ≈ 30.5 experts
```

Splitting the 3B active params into a fixed part (~1.97B: attention, embeddings, shared expert) and a per-expert part (~129M), that gives:

```
effective active ≈ 1.97B + 30.5 × 0.129B ≈ 5.9B
```

**5.9B predicted vs. 6.3B measured — a ~6% gap.** Feeding 5.9B back into the roofline model predicts **69.4 tok/s** against the actual **65.9 tok/s**, also within ~5%.

**A prediction built purely from published routing statistics — no throughput numbers involved — lands within 5% of reality.** The 4.2× MoE advantage isn't a mystery; it's exactly what expert reuse at small batch predicts.

## Can you actually make money doing this?

Back to the question that started this: is a DGX Spark a profitable way to sell inference, or just an expensive way to burn electricity?

**Cost of running the box, 24/7:**

|           | Value                                                                                                                                                                                                                             |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hardware  | £4,959.98, amortized over 3–5 years → **£0.11–£0.19/hr**                                                                                                                                                                          |
| Power     | 140 W SoC TDP – 240 W system rated (from the hardware table above) → **~£0.04–£0.07/hr** at [UK business electricity rates](https://connection-technologies.co.uk/blog/business-electricity-prices-per-kwh-uk-2026) (~£0.274/kWh) |
| **Total** | **~£0.15–£0.25/hr**                                                                                                                                                                                                               |

**Market rate for comparably-sized open models**, from [DeepInfra's pricing](https://deepinfra.com/pricing) — a reasonable stand-in for what this hardware could realistically charge:

|                           | Input $/M tok | Output $/M tok |
| ------------------------- | ------------- | -------------- |
| Dense-class (Qwen3-32B)   | $0.08         | $0.28          |
| MoE-class (Qwen3-30B-A3B) | $0.12         | $0.50          |

Notably, MoE is priced _higher_ per output token despite being far cheaper to serve — the market prices on output quality, not on compute cost.

### At the throughput actually measured (batch=4)

|                                 | Dense      | MoE        |
| ------------------------------- | ---------- | ---------- |
| Output throughput               | 15.7 tok/s | 65.9 tok/s |
| Revenue/hr (output tokens only) | ~£0.013    | ~£0.094    |
| Cost/hr                         | ~£0.20     | ~£0.20     |
| **Revenue as % of cost**        | **~6%**    | **~47%**   |

At the configuration this post actually benchmarked, both models lose money every hour the box runs — dense badly, MoE less badly but still solidly underwater.

### Concurrency, realistically

Batch=4 with short prompts barely touches the KV cache (1.6–3.2% peak). Extrapolating the validated roofline model to where the pool fills — untested, not benchmarked — gives dense ~99 tok/s at ~46 concurrent requests (~£0.079/hr) and MoE ~240 tok/s at ~113 concurrent (~£0.34/hr).

Treating those as real revenue is wrong twice over. First, they require _sustaining_ that concurrency near 24/7, which a single box competing with hyperscale providers won't see — real traffic is bursty. At a more realistic 20–30% utilization, MoE's revenue drops to **~£0.07–£0.10/hr** and dense's to **~£0.016–£0.024/hr**, both below the ~£0.15–£0.25/hr cost floor. Second, they assume every request matches the benchmark's short 4,096-token prompt — a fraction of either model's 262,144-token native context — while real prompts nearer that context would eat far more KV cache per request, capping concurrency, and revenue, even lower.

**Bottom line: neither model works under realistic utilization.** Dense's per-token economics are too expensive at any batch size. MoE only clears cost in the unrealistic best case — full concurrency, sustained near-100% of the time — and sinks back underwater the moment utilization drops to what a single-box operator would actually see. This box is a better tool for avoiding paying per-token elsewhere than for reselling inference as a business — a single desktop can't match the hardware-utilization economics of hyperscale serving.
