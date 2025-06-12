---
layout: distill
title: "The Impact of GPU Networking (Part 1)"
# permalink: /main/
description: "This post examines the impact of GPU networking on transformer model training performance using Distributed Data Parallel (DDP), comparing high-speed intra-node NVLink with slower inter-node InfiniBand."
date: 2025-03-14
future: true
htmlwidgets: true
hidden: false

section_number: 1

previous_section_url: ".."
previous_section_name: "Introduction"

next_section_url: "../net_2"
next_section_name: "Networking Part 2"

giscus_comments: false

authors:
  - name: Tomas

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "Initial Hypotheses"
  - name: "Prior Knowledge"
  - name: "Experimental Setup"
  - name: "Experiments"
  - name: "Analysis"
  - name: "Communication Benchmark Experiments"
  - name: "Conclusions"
  - name: "Next Steps"
  - name: "Additional Information"

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

The question of how much GPU interconnects influence training performance in large transformer-based models recently sparked a lively discussion with a colleague. Specifically, we debated whether, for simpler parallelism approaches—namely distributed data parallelism (DDP)—network bandwidth and latency still affect performance as strongly as they do for more communication-heavy strategies, such as model parallelism.

In data parallelism, each GPU independently processes a slice of the overall batch, requiring gradients to be synchronized at each training step. Because this synchronization involves only the transfer of gradients—rather than splitting the entire model across devices, as in model parallelism—it is tempting to assume that the interconnect between GPU nodes might not be a major bottleneck.

To gain empirical insights, I conducted a series of experiments using PyTorch Lightning and DDP. In this post, I document my process, present the experimental results, and reflect on what they reveal about GPU interconnects in multi-GPU training. While this is not a formal research paper, I hope it serves as a helpful exploration of how networking can affect large-scale transformer training.

## Initial Hypotheses

- **H1**: For DDP, the interconnect between GPUs or GPU nodes is not a significant factor in training time for large transformer models.

- **H2**: When training transformer models with DDP, scaling up the number of GPUs results in near-linear speedups.

My prior experience with both training transformer-based models and running large-scale HPC simulations made me skeptical of these hypotheses. However, I wanted to test them rigorously rather than rely on intuition or anecdotal evidence.

## Prior Knowledge

In previous work published on [Zenodo](https://zenodo.org/records/13349541), I evaluated different HPC configurations for training GPT-2 and explored various parallelism strategies. Those findings hinted that:

- Some HPC services in the UK exhibit differences in setup that lead to noticeable variation in training performance.
- DDP can show near-linear scaling in certain scenarios, but network details matter once the model size grows large.

I set out to expand on these observations here by running new experiments in a more controlled setting.

### What is GPT-2?

GPT-2 is a transformer-based language model designed for text generation. Its architecture leverages self-attention mechanisms, allowing it to effectively model contextual relationships in sequences of text. In my experiments, I worked with two variants:

- GPT-2: ~85 million parameters
- GPT-2 Large: ~708 million parameters

These models were trained in BF16 precision, with a batch size of 16, on the Shakespeare dataset. The training code relies on [lightning-GPT](https://github.com/tomaslaz/lit-GPT), which wraps the [minGPT](https://github.com/karpathy/minGPT) implementation in PyTorch Lightning to simplify multi-GPU experiments.

### Distributed Data Parallelism (DDP)

[PyTorch Lightning](https://lightning.ai/docs/pytorch/1.9.5/) is a lightweight PyTorch wrapper for high-performance AI research, designed to abstract most boilerplate code and provide a high-level interface for training models. lightning-GPT enables training on multiple GPUs and nodes, supporting different parallelism strategies; for this work, I focus only on [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).

When using DDP, gradient synchronization across GPUs happens every step when a batch is processed. The batch is split across the GPUs into smaller portions (mini-batches). Each GPU processes a different mini-batch of data in parallel during each training step. After computing gradients locally on their portion of the data, GPUs synchronize these gradients using an all-reduce operation to ensure consistency across the entire model.

## Experimental Setup

The experiments were conducted on [Baskerville](https://www.baskerville.ac.uk/), a GPU cluster at the University of Birmingham. Each node contains four NVIDIA A100 GPUs (40GB/80GB) connected via NVLink 3.0, and nodes are interconnected with InfiniBand HDR (see Appendix A for system specifications).

### Measurement Approach

I recorded the time taken to complete the second epoch of training. This approach helps:

- Avoid initialization overheads in the first epoch.
- Provide a realistic sense of how sustained training performance scales with different hardware configurations.

The second epoch timing captures the pure training speed once the framework has loaded the data and established the initial model states.

## Experiments

### GPT-2 Experiments

I first tested the smaller GPT-2 model (~85M parameters) with different node/GPU configurations:

**Results**:

| Experiment | Model | Parallelism | Nodes | GPUs per Node | Epoch 2 time (s) | Peak memory (MiB) |
| ---------- | ----- | ----------- | ----- | ------------- | ---------------- | ----------------- |
| e-15-1     | gpt2  | DDP         | 1     | 1             | 157.60           | 3,687.91          |
| e-15-2     | gpt2  | DDP         | 1     | 2             | 62.96            | 3,693.55          |
| e-15-3     | gpt2  | DDP         | 2     | 1             | 62.55            | 3,696.01          |

**Observations**:

- As expected, using more GPUs decreases training time.
- For the smaller GPT-2 model, there is no significant difference whether both GPUs are on the same node or on different nodes.
- These results initially seem to support H1 (reduced importance of interconnect for small models) and H2 (strong scaling with an additional GPU).

### GPT-2 Large Experiments

Next, I repeated the procedure with GPT-2 Large (~708M parameters):

**Results**:

| Experiment | Model      | Parallelism | Nodes | GPUs per Node | Epoch 2 time (s) | Peak memory (MiB) |
| ---------- | ---------- | ----------- | ----- | ------------- | ---------------- | ----------------- |
| e-15-4-1   | gpt2-large | DDP         | 1     | 1             | 498.37           | 24,258.05         |
| e-15-5-3   | gpt2-large | DDP         | 1     | 2             | 266.83           | 24,258.38         |
| e-15-6-2   | gpt2-large | DDP         | 2     | 1             | 406.12           | 24,297.53         |

**Observations**:

- Again, adding more GPUs reduces training time, but the effect is more complex than with the smaller model.
- When the GPUs are located on different nodes (Experiment e-15-6-2), the epoch time is substantially slower than when both GPUs are on the same node (e-15-5-3).
- These results suggest that interconnect performance starts to matter significantly for larger models, contradicting both H1 and H2.

## Analysis

For GPT-2 Large, the number of steps per epoch differs based on the number of GPUs:

- Single GPU (e-15-4-1): 2,216 steps
- Two GPUs (e-15-5-3 and e-15-6-2): 1,108 steps

The theoretical peak transfer speeds are:

- NVLink 3.0: 600 GB/s
- InfiniBand HDR: 200 Gbps (25 GB/s)

Model Size Calculation:

- GPT-2 Large has 708 million parameters.
- Model size in BF16 precision: 708 _ 10^6 _ 16 bits / 8 bits per byte = 1.416 GB

Total Data Transfer per Epoch:

- For 1,108 steps, and considering the all-reduce operation requires two transfers per step:
- Total data transferred: 2 _ 1.416 GB _ 1,108 = 3.14 TB

Communication Time Estimates:

- NVLink 3.0: 3.14 TB / 600 GB/s = 5.23 s
- InfiniBand HDR: 3.14 TB / 25 GB/s = 125.6 s

Comparing to Measured Times:

- Half of Single GPU Time: 498.37 s / 2 = 249.19 s.
  - This illustrates the ideal case where the training time halves with the addition of a second GPU assuming no communication overhead.
- Estimated Time with NVLink 3.0: 249.19 s + 5.23 s = 254.42 s
  - Align reasonably well with the observed timing in e-15-5-3: 266.83 s
- Estimated Time with InfiniBand HDR: 249.19 s + 125.6 s = 374.79 s
  - Align reasonably well with the observed timing in e-15-6-2: 406.12 s

Note: The discrepancies are likely due to factors like latency, network overhead, and resource contention.

## Communication Benchmark Experiments

To isolate the communication overhead, I wrote a small benchmark [script](../assets/scripts/bench_allreduce.py) measuring the throughput of a distributed all-reduce on 700M parameters in BF16 (~1.32 GB). The script relies on PyTorch’s collective communication (NCCL or Gloo) and reports the effective transfer speed:

**Results**:

| Experiment | Nodes | GPUs per Node | Backend          | Protocol | Measured Transfer speed (GB/s) | Time (s) |
| ---------- | ----- | ------------- | ---------------- | -------- | ------------------------------ | -------- |
| e-16-1     | 1     | 2             | NVLink 3.0       | NCCL     | 159.29                         | 0.02     |
| e-16-1     | 1     | 2             | PCIe 4.0         | NCCL     | 26.90                          | 0.10     |
| e-16-2     | 2     | 1             | InfiniBand (HDR) | NCCL     | 21.67                          | 0.12     |
| e-16-2     | 2     | 1             | Ethernet (25GbE) | NCCL     | 3.01                           | 0.88     |
| e-16-2     | 2     | 1             | InfiniBand (HDR) | Gloo     | 1.60                           | 1.70     |
| e-16-2     | 2     | 1             | Ethernet (25GbE) | Gloo     | 1.60                           | 1.65     |

**Observations**:

- NVLink (intra-node) throughput is orders of magnitude higher than inter-node InfiniBand.
- Times align with the differences observed in full training experiments: communication overhead grows substantially once multiple nodes are involved and model size is large.

### Recalculating Training Time Differences:

For e-15-5-3 (NVLink 3.0), additional time due to communication:

- 1,108 \* 0.02 s = 22.16 s
- Total estimated time: 249.19 s + 22.16 s = 271.35 s
- Close to measured time: 266.83 s

For e-15-6-2 (InfiniBand HDR), additional time due to communication:

- 1,108 \* 0.12 s = 132.96 s
- Total estimated time: 249.19 s + 132.96 s = 382.15 s
- Close to measured time: 406.12 s

Again, discrepancies are due to other factors affecting training time.

## Conclusions

These experiments demonstrate that as transformer models grow larger, the significance of GPU interconnects becomes harder to ignore—even in Distributed Data Parallel setups. For smaller models like GPT-2 (~85M parameters), interconnect speed has minimal effect, in line with initial hypotheses H1 and H2. However, GPT-2 Large (~708M parameters) reveals a striking gap between intra-node NVLink performance and inter-node InfiniBand.

In short:

- H1 and H2 are not supported by the data.
- Interconnect efficiency matters significantly for large-scale DDP workloads.
- Adding more GPUs does not guarantee near-linear speedups if inter-node bandwidth lags behind intra-node connectivity.

## Next Steps

In the [Part 2](../net_2), I will extend these experiments to 4 GPUs spread across 1, 2, and 4 nodes. This will shed further light on how network topologies (e.g., ring versus tree) influence performance when the number of GPUs per node varies.

# Additional Information

## Appendix A: Baskerville HPC System Specifications

The experiments were conducted on the [Baskerville](https://www.baskerville.ac.uk/) HPC system, which has the following specifications:

### Compute Nodes

There are 57 SD650-N V2 liquid-cooled compute trays with:

- 2× Intel® Xeon® Platinum 8360Y CPUs, each with 36 cores at 2.4GHz (boost up to 3.5GHz)
- 512GB RAM (16× 32GB DDR4)
- 1TB NVMe M.2 device (used for the OS and available as /scratch-local)
- 1× 25GbE NVIDIA® Mellanox® adapter (on-planar ConnectX-4 port)
- 1× HDR (200Gbps) NVIDIA Mellanox Infiniband port (ConnectX-6 PCIe Gen4 adapter)
- NVIDIA HGX-100 GPU planar
- 4× NVIDIA A100 GPUs

The GPUs on 11 nodes have 80GB RAM, while those on the remaining 46 nodes have 40GB RAM. The GPUs are interconnected using NVIDIA NVLINK.

### Network

Baskerville uses three networks:

- An isolated 1GbE management network (not user-accessible)
- A 25GbE high-speed Ethernet network built using NVIDIA Mellanox Spectrum®-2 switches running NVIDIA Cumulus® Linux
- An HDR fat-tree InfiniBand network built using NVIDIA Mellanox Quantum HDR switches (QM8790) and ConnectX-6 PCIe gen 4 adapters which provides a full 200Gb network connection

### Storage

The system is equipped with Lenovo DSS-G storage systems running IBM® Spectrum Scale™:

- 1× DSS-G250 equipped with 418× 16TB HDD
- 1× DSS-G204 equipped with 48× 7.68TB SSD
