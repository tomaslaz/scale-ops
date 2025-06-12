import torch
import torch.distributed as dist
import os
import time
import logging


def get_local_rank():
    """
    Get the local rank of the current process.

    Returns:
        int: The local rank of the current process.
    """
    # First, check if torchrun set LOCAL_RANK
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    # If not, check SLURM variables
    elif 'SLURM_LOCALID' in os.environ:
        return int(os.environ['SLURM_LOCALID'])
    elif 'SLURM_PROCID' in os.environ:
        # Use SLURM_PROCID modulo GPUs per node.
        gpus_per_node = int(os.environ.get('GPUS_PER_NODE', torch.cuda.device_count()))
        return int(os.environ['SLURM_PROCID']) % gpus_per_node
    else:
        # Default to 0 if no environment variable is set
        return 0  # Default to 0


def init_distributed(backend="nccl", init_method="env://"):
    """
    Initialise the distributed environment.

    Args:
        backend (str): The backend to use for distributed communication.
            Choose 'nccl' for multi-GPU setups with NVIDIA GPUs.
            Choose 'gloo' for CPU training or when GPUs are not NVIDIA.
            Choose 'mpi' if your environment is already set up with MPI.
        init_method (str): The initialization method for distributed communication.
            'env://' is convenient in environments where you can set environment variables.
            'file://' requires a shared filesystem accessible by all nodes.
            'tcp://' requires that the specified port is open and accessible across all nodes.
    """

    logging.debug(f"Initialising distributed environment. Backend: {backend}, init_method: {init_method}")

    # Check and validate environment variables if using 'env://' method
    if init_method == "env://":
        required_vars = ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"]
        missing_vars = [var for var in required_vars if var not in os.environ]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {', '.join(missing_vars)}")

        # Validate MASTER_PORT
        try:
            master_port = int(os.environ["MASTER_PORT"])
            if not (0 <= master_port <= 65535):
                raise ValueError("MASTER_PORT must be an integer between 0 and 65535.")
        except ValueError as e:
            raise ValueError(f"Invalid MASTER_PORT: {e}")

        # Log environment variables
        logging.debug(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        logging.debug(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
        logging.debug(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}")
        logging.debug(f"Global Rank (RANK): {os.environ['RANK']}")

        if "LOCAL_RANK" in os.environ:
            logging.debug(f"Local Rank (LOCAL_RANK): {os.environ['LOCAL_RANK']}")
        else:
            logging.warning("LOCAL_RANK not set.")

    # Initialize the process group
    if dist.is_initialized():
        logging.warning("Distributed environment already initialised. Skipping initialization.")
        return
    try:
        dist.init_process_group(backend=backend, init_method=init_method)
        if not dist.is_initialized():
            raise RuntimeError("Failed to initialise the distributed process group.")
        logging.info("Distributed process group initialised successfully.")
    except Exception as e:
        logging.error(f"Failed to initialise the distributed process group: {e}")
        raise


def measure_allreduce_speed(tensor_size, experiments=10, calls=10, dtype="bfloat16", device=None, local_rank=None):
    """
    Measure the speed of all-reduce operations, discounting the computation time.

    Args:
        tensor_size (int): Number of elements in the tensor (of specified dtype).
        experiments (int): Number of experiments to run.
        calls (int): Number of all-reduce calls per experiment.
        dtype (str): Data type of the tensor.
        device (torch.device): The device to run the tensor on.
        local_rank (int): The local rank of the process.
    """
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        logging.info(f"Reporting: rank: {rank}, world_size: {world_size}")

    with torch.no_grad():
        # Create a tensor to reduce
        tensor_dtype = getattr(torch, dtype)
        tensor = torch.rand(tensor_size, dtype=tensor_dtype, device=device)

    # Optionally check for available GPU memory
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(device).total_memory
        required_memory = tensor.element_size() * tensor.nelement()
        if required_memory > total_memory:
            raise MemoryError("Tensor size exceeds available GPU memory.")

    dist.barrier()

    # Warm-up
    for _ in range(5):
        dist.all_reduce(tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed_time = 0
    computation_time = 0

    for exp_i in range(experiments):
        dist.barrier()
        
        if rank == 0:
            logging.info(f"Experiment {exp_i + 1}/{experiments}...")

        # Measure computation time
        with torch.no_grad():
            comp_tensor = tensor.clone()
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            comp_start = time.time()
            for _ in range(calls):
                # Simulate element-wise reduction locally
                comp_tensor += comp_tensor  # Or use appropriate reduction operation
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            comp_end = time.time()

            computation_time += (comp_end - comp_start) / calls

        # Measure total all-reduce time
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(calls):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.time()

        # Calculate total time
        elapsed_time += (end - start) / calls

        # Rest between experiments
        time.sleep(1)

    # Average times over experiments
    elapsed_time /= experiments
    computation_time /= experiments

    # Calculate communication time
    communication_time = elapsed_time - computation_time
    if communication_time < 0:
        communication_time = 0  # Adjust for any negative values due to measurement inaccuracies

    tensor_size_mb = tensor_size * tensor.element_size() / (1024 ** 2)
    # Total data communicated across all processes
    total_data_all_processes_mb = tensor_size_mb * 2 * (world_size - 1)

    # Calculate communication speed
    if communication_time > 0:
        speed_all_processes = total_data_all_processes_mb / communication_time
    else:
        speed_all_processes = float('inf')  # Indicate extremely high speed due to negligible communication time

    if rank == 0:
        logging.info(f"The results are averaged over {experiments} experiments with each making {calls} all reduce calls.")

        logging.info(
            f"Avg All-Reduce Elapsed Time (including computation time): {elapsed_time:.6f} seconds"
        )
        logging.info(
            f"Avg All-Reduce Theoretical Computation Only Time: {computation_time:.6f} seconds"
        )
        logging.info(
            f"Avg All-Reduce Communication (elapsed - computation) Time: {communication_time:.6f} seconds"
        )

        logging.info(
            f"Total Data Communicated Across All Processes: {total_data_all_processes_mb:.2f} MB "
            f"in {communication_time:.6f} seconds, Communication Speed: {speed_all_processes:.2f} MB/s"
        )


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--backend", type=str, default="nccl",
                        help="Backend for distributed communication (default: nccl)")
    parser.add_argument("--init-method", type=str, default="env://",
                        help="Initialization method for distributed communication (default: env://)")
    parser.add_argument("--tensor-size", type=int, default=708_000_000,
                        help="Size of the tensor in elements of specified dtype (default: 708_000_000)")
    parser.add_argument("--experiments", type=int, default=10,
                        help="Number of experiments to run (default: 10)")
    parser.add_argument("--calls", type=int, default=10,
                        help="Number of all-reduce calls per experiment (default: 10)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Data type of the tensor (default: bfloat16)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    # Initialize local rank and device
    local_rank = get_local_rank()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    logging.debug(f"Get local rank. Local rank: {local_rank}, Device: {device}")

    # Initialize distributed communication
    init_distributed(backend=args.backend, init_method=args.init_method)

    try:
        # Measure all-reduce speed
        measure_allreduce_speed(
            tensor_size=args.tensor_size,
            experiments=args.experiments,
            calls=args.calls,
            dtype=args.dtype,
            device=device,
            local_rank=local_rank
        )
    finally:
        # Finalize
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
