import torch.distributed as dist
import numpy as np
import time
import json
import torch
import os
import socket
import argparse


def in_mb(size):
    return size / 1024**2


def _ranks_on_same_node(rank, world_size):
    hostname = socket.gethostname()
    hostname_length = torch.IntTensor([len(hostname)])
    dist.all_reduce(hostname_length, op=dist.reduce_op.MAX)
    max_hostname_length = hostname_length.item()

    encoding = [ord(c) for c in hostname]
    encoding += [-1 for c in range(max_hostname_length - len(hostname))]
    encoding = torch.IntTensor(encoding)

    all_encodings = [torch.IntTensor([0] * max_hostname_length) for _ in range(world_size)]
    dist.all_gather(all_encodings, encoding)

    all_encodings = [ec.numpy().tolist() for ec in all_encodings]
    counter = 0
    for i in range(rank):
        if all_encodings[rank] == all_encodings[i]:
            counter += 1
    return counter


def to_device(tensor, rank, world_size):
    """Place tensor to a gpu."""
    num_gpus_on_device = torch.cuda.device_count()
    assigned_id = _ranks_on_same_node(rank, world_size)
    return tensor.to('cuda:{}'.format(assigned_id))


def create_vector(length):
    """Create a torch vector of certain length."""
    v = torch.ones(length)
    size = in_mb(v.numpy().nbytes)
    return v, size


def main_mpi(n_iters, length, foutput):
    dist.init_process_group('mpi')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Create a vector for communication
    v, size = create_vector(length)
    v = to_device(v, rank, world_size)
    if rank == 0:
        print("Vector Size={:.6f} MB".format(size))

    # Measuring the time on P2P communication (on the receive end.)
    elapsed_time = []
    for i in range(n_iters):
        # Waits for all kernels in all streams on current device to complete
        torch.cuda.synchronize()

        # Synchronizes all processes.
        dist.barrier()
        t0 = time.time()
        dist.all_reduce(tensor=v, op=dist.reduce_op.SUM)

        # Waits for all kernels in all streams on current device to complete
        torch.cuda.synchronize()
        elapsed_time.append(time.time() - t0)

        if rank == 0:
            print("All reduce vector: {}-th iteration".format(i))

    if rank == 0:
        mean_time = sum(elapsed_time) / n_iters
        throughput = size / mean_time
        output = {"elapsed": elapsed_time,
                  "size": size, "n_iters": n_iters, "length": length}

        if os.path.exists(foutput):
            os.remove(foutput)

        with open(foutput, 'w') as f:
            json.dump(output, f)

        print("Transfer {:.6f} MB in {:.6f} seconds. [{:.6f} MB/s]"
              .format(size, mean_time, throughput))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""""")
    parser.add_argument('--n_iters', type=int, help='number of loops to repeat.')
    parser.add_argument('--length', type=int, help='length of float32 vectors to transfer.')
    parser.add_argument('--foutput', type=str, help='name of output file.')
    args = parser.parse_args()

    main_mpi(args.n_iters, args.length, args.foutput)
