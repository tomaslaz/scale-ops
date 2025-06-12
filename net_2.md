---
layout: distill
title: "The Impact of GPU Networking (Part 2)"
# permalink: /main/
description: "Part 2 builds on earlier experiments by examining how distributing 4 GPUs across 1, 2, and 4 nodes impacts transformer model training, with a focus on network topology and NIC sharing."
date: 2025-03-14
future: true
htmlwidgets: true
hidden: false

section_number: 2

previous_section_url: "../net_1"
previous_section_name: "Networking Part 1"

next_section_url: ""
next_section_name: "..."

giscus_comments: false

authors:
  - name: Tomas

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Particularities of Distributed Training using 4 GPUs on 1, 2, and 4 Nodes
  - name: GPT-2 Large Experiments
  - name: The basics
  - name: Analysis
  - name: Key Takeaways (TL;DR)
  - name: Additional Information

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

[In Part 1](../net_1), I carried out a series of experiments to highlight the significance of GPU interconnects on the training performance of transformer-based models using Distributed Data Parallelism (DDP). These experiments demonstrated that even for relatively modest models, such as GPT-2 Large (700M parameters), the efficiency of the interconnect between GPUs or GPU nodes plays an important role in overall training time, even when data parallelism is used.

Building upon these insights, in Part 2 I delve a bit deeper into the impact of networking when more GPU nodes are used in distributed training. I conduct a series of experiments using 4 GPUs distributed across 1, 2, and 4 nodes to illustrate how network topology and interconnects between nodes affect the training performance of transformer models.

## Particularities of Distributed Training using 4 GPUs on 1, 2, and 4 Nodes

To begin, I ran a series of experiments using 4 GPUs distributed across 1, 2, and 4 nodes, and as in the Part 1, I used the [Baskerville](https://www.baskerville.ac.uk/) HPC system (see Appendix A for system specifications). The experiments were conducted using the same GPT-2 Large model and training script as in Part 1.

### GPT-2 Large Experiments

| Experiment | Model      | Parallelism | Nodes | GPUs per Node | Epoch 2 time (s) |
| ---------- | ---------- | ----------- | ----- | ------------- | ---------------- |
| e-15-7-5   | gpt2-large | DDP         | 1     | 4             | 135.42           |
| e-15-8-5   | gpt2-large | DDP         | 2     | 2             | 895.85           |
| e-15-9-4   | gpt2-large | DDP         | 4     | 1             | 251.68           |

These results not only contradict H1 and H2 from Part 1—since the training times between `e-15-7-5` and `e-15-9-4` are almost double—but also show that intermediate configurations, such as `e-15-8-5` (2-node, 2-GPU-per-node), exhibit significantly slower training times compared to both `e-15-7-5` and `e-15-9-4`.

To ensure that these results are not anomalies, I reused the same Python [script](../assets/scripts/bench_allreduce.py) from Part 1 to benchmark the performance of the all-reduce operation in a distributed PyTorch environment. As before, the script uses 700M parameters (GPT-2 Large) in BF16 precision (1.32 GB) and measures the transfer speed of the all-reduce operation between GPUs. The results are as follows:

| Experiment | Nodes | GPUs per Node | Measured Transfer speed (GB/s) | Time (s) |
| ---------- | ----- | ------------- | ------------------------------ | -------- |
| e-16-4-1   | 1     | 4             | 875.1                          | 0.009    |
| e-16-4-4   | 2     | 2             | 25.6                           | 0.309    |
| e-16-4-3   | 4     | 1             | 50.0                           | 0.158    |

_Note: The total data transferred per experiment in Table 2 is around 7.9 GB—that is, 1.32 GB transferred back and forth three times (N - 1)._

---

**Observation 1:** The results from the all-reduce operation are consistent with the training times of the GPT-2 Large model and show that 2-node, 2-GPU-per-node configuration has a much slower transfer speed compared to the 1-node, 4-GPU-per-node or 4-node, 1-GPU-per-node configurations.

---

This is somewhat counterintuitive, as one might expect that the transfer speed between 2 nodes with 2 GPUs per node would be faster than that between 4 nodes with 1 GPU per node, given that intra-node communication is clearly faster than inter-node communication. Thus, having fewer nodes should be more efficient. However, the results show that this is not the case.

Let's try to understand why this is happening by starting with some communication basics for distributed training.

## The basics

In DDP, each GPU processes a portion of the data and computes gradients independently. After each iteration, these gradients need to be synchronized (averaged) across all GPUs. This synchronization requires communication between GPUs, and the efficiency of this communication directly affects overall training time.

To synchronize gradients, PyTorch uses the all-reduce operation—a collective communication operation that reduces the input tensor across all GPUs and returns the result to each GPU. Since I am using NVIDIA GPUs, the all-reduce operation is implemented with the NCCL library, which is optimized for NVIDIA GPUs and, when possible, uses the NVLink interconnect for communication between GPUs on the same node. The NCCL library also automatically detects the hardware topology and optimizes communication accordingly.

There are several topologies that NCCL can employ, such as ring, tree, double binary tree, multi-node, and hybrid. The choice of topology depends on the hardware configuration and the number of GPUs used during training, and it is automatically selected by NCCL.

### Ring topology

The ring topology is the most common topology used by NCCL. In this topology, GPUs are connected in a ring, with each GPU connected to two others. The ring topology is typically used when all GPUs are connected with fast interconnects, such as NVLink.

```
       Rank 0
      /     \
 Rank 1       Rank 2
      \     /
       Rank 3
```

The ring topology in the NCCL all-reduce operation is executed in two phases.

In the first phase, each GPU splits its data into equal-sized chunks and initiates a reduce-scatter operation. Each rank sends one of its chunks to its neighbor while simultaneously receiving the corresponding chunk from the previous rank, adding the received values to its own. This exchange is performed over P–1 steps (with P being the total number of GPUs), ensuring that each chunk is progressively reduced across all nodes.

In the second phase, an all-gather operation circulates these reduced chunks around the ring so that every rank eventually collects all chunks and obtains the complete, fully reduced result.

### Tree topology

The tree topology is typically used when GPUs are not connected with NVLink and is more complex than the ring topology. In this topology, GPUs are connected in a tree structure, with each GPU connected to one or more others. The tree topology is typically used when GPUs are connected via InfiniBand.

```
    Rank 0
   /     \
Rank 1   Rank 2
           |
         Rank 3
```

### Double binary tree

This enhanced topology uses two distinct binary trees — one for scattering (broadcast) and one for gathering (reduction) — to optimize bandwidth utilization.

In this setup, one tree manages the reduction phase by aggregating data from all ranks into a single root (the "up" tree), while the other handles the broadcasting phase by distributing the aggregated result back to all ranks (the "down" tree).

The reduction tree is identified by its final node having no children, as it collects data from every node, whereas the broadcast tree starts at a top node without a parent, initiating the dissemination process.

### Hybrid

Hybrid topologies allow NCCL to integrate multiple network structures — such as a ring within each node and a tree across nodes—tailoring the communication pattern to the underlying physical network. NCCL automatically detects and configures the best topology based on available hardware, though users can sometimes influence the selection through environment variables.

For example, within a node, GPUs might be connected via NVLink forming a ring, while nodes are interconnected via Infiniband or Ethernet forming a tree. This approach employs a two-level design, first optimizing intra-node communication, then inter-node connectivity, which generally exhibits lower latency within nodes and higher latency between nodes.

## Analysis

Let's see which topology NCCL has detected for the `e-15-7-5`, `e-15-8-5`, and `e-15-9-4` experiments by adding `export NCCL_DEBUG=INFO` to the job submission script.

The NCCL logs indicate that in every example, NCCL uses a combination of ring and tree topologies—a hybrid approach in which the best communication strategy is dynamically selected.

The logs for the tree topology are given in the following format:

```
Trees P1/P2/P3->R->C
```

where P is the parent(s) ranks (-1 if none), R is the current rank, and C is the child's rank (-1 if none).

In this context, parent in a reduction operation collects data from the child ranks, and in a broadcasting operation sends data to the child ranks, whereas the child does the opposite.

Let's look into the topologies of the experiments in more detail.

### 1 Node, 4 GPUs

In this case, all GPUs are on the same node which is NVLink connected. In this particular case, NCCL establishes 24 parallel communication “pipelines” (channels) to communicate between the GPUs. The high number of channels allows for a high level of parallelism, to make the most of the available NVLink bandwidth.

Each channel sets up its own independent communication context, i.e., each one builds its own ring topology for point-to-point communication and, when applicable, its own tree structure for certain collective operations (like reductions). This design allows NCCL to overlap communication and computation across multiple channels, maximizing throughput and efficiency on multi-GPU systems.

In this case, NCCL uses 24 ring and tree topologies, one for each channel, even though some of them are identical or may not be used.

Rings:

```
NCCL INFO Channel 00/24 :    0   1   2   3
NCCL INFO Channel 01/24 :    0   1   3   2
NCCL INFO Channel 02/24 :    0   2   3   1
...
NCCL INFO Channel 06/24 :    0   1   2   3
NCCL INFO Channel 07/24 :    0   1   3   2
...
NCCL INFO Channel 23/24 :    0   3   2   1
```

Trees:

```
NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1 [2] 1/-1/-1->0->-1 [3] 1/-1/-1->0->-1 [4] 2/-1/-1->0->-1 [5] 2/-1/-1->0->-1 ... [23] 3/-1/-1->0->1

NCCL INFO Trees [0] 2/-1/-1->1->0 [1] 2/-1/-1->1->0 [2] 2/-1/-1->1->0 [3] 2/-1/-1->1->0 [4] 3/-1/-1->1->2 [5] 3/-1/-1->1->2 ... [23] 0/-1/-1->1->-1

NCCL INFO Trees [0] 3/-1/-1->2->1 [1] 3/-1/-1->2->1 [2] 3/-1/-1->2->1 [3] 3/-1/-1->2->1 [4] 1/-1/-1->2->0 [5] 1/-1/-1->2->0 ... [23] -1/-1/-1->2->3

NCCL INFO Trees [0] -1/-1/-1->3->2 [1] -1/-1/-1->3->2 [2] -1/-1/-1->3->2 [3] -1/-1/-1->3->2 [4] -1/-1/-1->3->1 [5] -1/-1/-1->3->1 ... [23] 2/-1/-1->3->0
```

### 2 nodes, 2 GPUs per node

It is important to note that Rank 0 and Rank 1 are on the same node, and Rank 2 and Rank 3 are on the other node. In this case only 2 channels are established, and as in the previous case, NCCL uses a combination of ring and tree topologies in each channel.

Rings:

```
NCCL INFO Channel 00/02 :    0   1   2   3
NCCL INFO Channel 01/02 :    0   1   2   3
```

Double binary trees:

```
NCCL INFO Trees [0] -1/-1/-1->3->2 [1] -1/-1/-1->3->2
NCCL INFO Trees [0] 1/2/-1->0->-1 [1] 1/-1/-1->0->2
NCCL INFO Trees [0] 3/-1/-1->2->0 [1] 3/0/-1->2->-1
NCCL INFO Trees [0] -1/-1/-1->1->0 [1] -1/-1/-1->1->0

Tree 0: Reduction tree

           rank 0
          /      \
     rank 1       rank 2
                   |
                  rank 3

Tree 1: Broadcast tree

            rank 2
          /      \
    rank 3        rank 0
                    |
                  rank 1
```

### 4 nodes, 1 GPU per node

It is important to note that all ranks are on different nodes.

Rings:

```
NCCL INFO Channel 00/02 :    0   1   2   3
NCCL INFO Channel 01/02 :    0   1   2   3
```

Double binary tree:

```
NCCL INFO Trees [0] -1/-1/-1->3->2 [1] 1/-1/-1->3->-1
NCCL INFO Trees [0] -1/-1/-1->1->2 [1] 2/0/-1->1->3
NCCL INFO Trees [0] 1/3/-1->2->0 [1] -1/-1/-1->2->1
NCCL INFO Trees [0] 2/-1/-1->0->-1 [1] -1/-1/-1->0->1

Tree 0: Reduction tree

           Rank 0
             |
           Rank 2
          /      \
    Rank 1        Rank 3

Tree 1: Broadcast tree

           Rank 3
             |
           Rank 1
          /      \
    Rank 0        Rank 2
```

### Ring or Tree?

Since the NCCL logs show that a combination of ring and tree topologies is being used in each of the experiments, it is difficult to say which topology is being used more or if the ring or tree topology is the bottleneck in the experiments. However, it is possible to set the topology manually using the `NCCL_ALGO` environment variable by setting it to `RING` or `TREE` to force the use of the ring or tree topology respectively and compare the results.

| Experiment | Nodes | GPUs per Node | Topology | Measured Transfer speed (GB/s) | Time (s) |
| ---------- | ----- | ------------- | -------- | ------------------------------ | -------- |
| e-16-5-1-r | 1     | 4             | Ring     | 922.4                          | 0.009    |
| e-16-5-1-t | 1     | 4             | Tree     | 606.6                          | 0.013    |
| e-16-5-2-r | 2     | 2             | Ring     | 13.5                           | 0.584    |
| e-16-5-2-t | 2     | 2             | Tree     | 25.6                           | 0.309    |
| e-16-5-3-r | 4     | 1             | Ring     | 53.9                           | 0.158    |
| e-16-5-3-t | 4     | 1             | Tree     | 45.0                           | 0.158    |

---

**Observation 2:** The results show that the ring topology is faster than the tree topology for the 1-node, 4-GPU-per-node and 4-node, 1-GPU-per-node configurations. However, the ring topology is slower than the tree topology for the 2-node, 2-GPU-per-node configuration.

---

### Why Is the 2‐Node, 2‐GPU‐per‐Node Configuration Slower?

Intuitively, the 2-node, 2-GPU-per-node configuration should be faster with a tree topology than the ring topology using 4 nodes with 1 GPU each, as the former has fewer nodes and relies less on inter-node communication. However, experiments consistently show the opposite outcome for both the GPT-2 Large model and a standalone PyTorch AllReduce operation.

The only explanation for this discrepancy is the hardware configuration of the Baskerville system, where the experiments were conducted. In this system, each node has one network interface card (NIC) for the InfiniBand HDR network, shared by all GPUs on that node. This means that when two GPUs are on the same node, they must share the single NIC, effectively reducing the per-GPU bandwidth for inter-node communication.

This can be clearly illustrated by comparing the `e-16-5-2-r` (2 nodes, 2 GPUs each) and `e-16-5-3-r`(4 nodes, 1 GPU each) experiments. The measured transfer speed for the 2-node, 2-GPU-per-node configuration is significantly slower—roughly four times slower than the 4-node setup. This is because `e-16-5-3-r` benefits from parallel bi-directional communication between the nodes, effectively doubling the bandwidth, whereas the 2-node configuration must share the NIC, effectively halving the bandwidth.

This is also why the tree topology is faster than the ring topology for the 2-node, 2-GPU-per-node configuration (`e-16-5-2-t` vs. `e-16-5-2-r`). In the tree topology, results are first aggregated on the node and then sent to the other node, reducing NIC contention and mitigating the bandwidth bottleneck.

## Key Takeaways (TL;DR)

Key takeaways from these experiments underscore the importance of GPU distribution, network interfaces, and communication topologies in multi-node training. In particular:

- Collocating all GPUs on a single node (and leveraging high-bandwidth NVLink) yield notably faster training than spreading GPUs across multiple nodes.
- While theory suggests that fewer nodes with more GPUs each might be preferable, real-world hardware constraints—like sharing a single NIC per node—can drastically reduce effective bandwidth.
- Forcing ring or tree topologies demonstrates that ring can excel when GPUs do not contend for inter-node communication, whereas tree may perform better when multiple GPUs share a single NIC.
- Ultimately, empirical tests in actual cluster environments, rather than relying solely on theoretical models, are essential to identify the best GPU distribution and communication strategy for large-scale transformer training.

Thank you for reading this far! I warmly welcome any feedback or discussion, especially if you spot potential oversights in my reasoning or experiments.

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
