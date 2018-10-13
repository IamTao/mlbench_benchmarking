import matplotlib.pyplot as plt
import json
import os
import numpy as np
import seaborn as sns


def read_one(raw_data_dir):
    files = sorted(os.listdir(raw_data_dir), key=lambda x: int(x))
    datas = []
    for file in files:
        with open(os.path.join(raw_data_dir, file), 'r') as f:
            datas.append(json.load(f))

    sizes = list(map(lambda x: x['size'], datas))
    throughput_mean = list(map(lambda x: np.mean(x['size'] / np.array(x['elapsed'][1:])), datas))
    throughput_std = list(map(lambda x: np.std(x['size'] / np.array(x['elapsed'][1:])), datas))
    # time_mean = list(map(lambda x: np.mean(x['elapsed']), datas))
    # time_std = list(map(lambda x: np.std(x['elapsed']), datas))
    # return sizes, time_mean, time_std
    return sizes, throughput_mean, throughput_std


def main():
    fig, ax = plt.subplots()

    nodes = ['nodes-8-gpus-1', 'nodes-4-gpus-2', 'nodes-2-gpus-4', 'nodes-1-gpus-8']
    colors = ['r', 'g', 'b', 'k']
    for color, node in zip(colors, nodes):
        for backend in ['mpi', 'nccl']:
            task = node + '/' + backend
            raw_data_dir = "output/" + task
            sizes, mean, std = read_one(raw_data_dir)

            line = '-' if 'mpi' in task else '--'
            fmt = color + 'o' + line
            ax.errorbar(sizes, mean, yerr=std, fmt=fmt, label=task)

    ax.plot([1e-4, 10**4], [950, 950], 'k--', alpha=0.3, label='Network Bandwidth')

    ax.set_xlim(1e-4, 10**4)
    ax.set_xscale('log')
    ax.set_xlabel('Vector Size [MB]')
    ax.set_ylabel('Throughput [MB/s]')
    ax.set_title("PyTorch-MPI All Reduce Benchmarks")
    ax.grid()
    ax.legend()
    fig.savefig("output/fig.png")


if __name__ == '__main__':
    main()
