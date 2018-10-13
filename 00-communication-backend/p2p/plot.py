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
    throughput_mean = list(map(lambda x: np.mean(x['Throughput (MB/s)'][1:]), datas))
    throughput_std = list(map(lambda x: np.std(x['Throughput (MB/s)'][1:]), datas))
    return sizes, throughput_mean, throughput_std


def main():
    fig, ax = plt.subplots()

    for task in ['cpu', 'gpu', 'cuda-ipc', 'no-cuda-ipc']:
        raw_data_dir = "output/" + task
        sizes, mean, std = read_one(raw_data_dir)
        ax.errorbar(sizes, mean, yerr=std, fmt='-o', label=task)

    ax.plot([1e-4, 10**4], [950, 950], 'k--', alpha=0.3, label='Network Bandwidth')

    ax.set_xlim(1e-4, 10**4)
    ax.set_xscale('log')
    ax.set_xlabel('Vector Size [MB]')
    ax.set_ylabel('Throughput [MB/s]')
    ax.set_title("PyTorch-MPI P2P Benchmarks")
    ax.grid()
    ax.legend()
    fig.savefig("output/fig.png")


if __name__ == '__main__':
    main()
