import torch.distributed as dist
import torch
import argparse
import os
import time
import json


def in_mb(size):
    return size / 1024**2


def create_vector(length):
    """Create a torch vector of certain length."""
    v = torch.ones(length)
    size = in_mb(v.numpy().nbytes)
    return v, size


def to_device(tensor, local_rank):
    """Place tensor to a gpu."""
    return tensor.cuda(local_rank)


def main(n_iters, length, foutput, local_rank):
    import torch.distributed as dist

    master_port = os.environ['MASTER_PORT']
    master_addr = os.environ['MASTER_ADDR']
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])

    os.environ['NCCL_DEBUG'] = 'INFO'
    # print("NCCL_DEBUG", os.environ['NCCL_DEBUG'])
    print("master_addr={} master_port={} world_size={} rank={}"
          .format(master_addr, master_port, world_size, rank))

    # by the specified environment.
    dist.init_process_group(backend="nccl", init_method='env://')

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    tensor_list = []
    for dev_idx in range(torch.cuda.device_count()):
        v, size = create_vector(length)
        v = to_device(v, dev_idx) * rank
        tensor_list.append(v)
    print("Vector created at rank {}".format(rank))

    if rank == 0:
        print("Vector Size={:.6f} MB".format(size))

    # Measuring the time on P2P communication (on the receive end.)
    elapsed_time = []
    for i in range(n_iters):
        # Waits for all kernels in all streams on current device to complete
        torch.cuda.synchronize()

        t0 = time.time()
        dist.all_reduce_multigpu(tensor_list=tensor_list, op=dist.reduce_op.SUM)

        torch.cuda.synchronize()
        elapsed_time.append(time.time() - t0)

        print("All reduce vector at rank {}: {}-th iteration".format(rank, i))

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
    parser.add_argument('--local_rank', type=str, help='name of output file.')
    args = parser.parse_args()

    main(args.n_iters, args.length, args.foutput, args.local_rank)
