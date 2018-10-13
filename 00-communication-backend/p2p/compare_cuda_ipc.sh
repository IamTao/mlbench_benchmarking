#!/bin/bash

# ------------------------------------------------------------------------------
# This script compares 1-node-2-GPUs p2p communication with or without CUDA-IPC
# ------------------------------------------------------------------------------

# ---------------------------- Cluster Configs -------------------------------
export CLUSTER_NAME=cuda-ipc
export RELEASE_NAME=rel
export MYVALUES_FILE=$(PWD)/myvalues.yaml
export NUM_NODES=1
export NUM_GPU_PER_NODE=2

DISK_SIZE=10GB
MACHINE_ZONE=europe-west1-b
GCE_PERSISTENT_DISK=gce-nfs-disk
# -----------------------------------------------------------------------------

function run_jobs(){
    . ../kubeutils.sh --source-only
    # Create directory for output
    local OUTPUT=$1
    local use_cuda_ipc=$2
    mkdir -p ${OUTPUT}

    # Gather the names of workers
    local wnames=($(kube::worker::hostnames))
    echo "Worker names:" + ${wnames[@]}

    # Copy the file for benchmarking to all workers.
    kube::scp_one ${PWD}/p2p.py /p2p.py

    # make mpirun command
    grid=( 10 100 1000 10000 100000 1000000 10000000 100000000 1000000000 )
    for length in ${grid[@]}
    do
        remote_output="/tmp/${length}"

        cmd="/.openmpi/bin/mpirun -n 2 $use_cuda_ipc /conda/bin/python /p2p.py --cuda --n_iters 100 --length ${length} --foutput ${remote_output}"
        kube::exec_one ${wnames[0]} ${cmd}

        # Download the time from the receiv end.
        kube::download_one ${wnames[0]} ${remote_output} ${OUTPUT}
    done
}

arg1=$1
case $arg1 in
    create-cluster )
        # Create a gcloud cluster
        bash ../../launch.sh create-gpu

        # Install mlbench onto the cluster
        bash ../../launch.sh install
        
        . ../kubeutils.sh --source-only

        # Fix the symbolic link problem
        # TODO: Solve this problem in docker image
        kube::scp_one ${PWD}/fix_link.sh /fix_link.sh
        kube::exec_all "bash /fix_link.sh"
        ;;

    cleanup-cluster )
        bash ../../launch.sh cleanup
        ;;

    no-cuda-ipc )
        OUTPUT=output/${arg1}
        run_jobs ${OUTPUT} "--mca btl_smcuda_use_cuda_ipc 0"
        ;;

    cuda-ipc )
        OUTPUT=output/${arg1}
        run_jobs ${OUTPUT} "--mca btl_smcuda_use_cuda_ipc 1"
        ;;
esac