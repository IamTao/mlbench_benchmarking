#!/bin/bash

# ---------------------------------------
# Compare 2-node-1-GPUs p2p communication
# ---------------------------------------

# ---------------------------- Cluster Configs -------------------------------`
export CLUSTER_NAME=cpu-gpu-p2p
export RELEASE_NAME=rel
export MYVALUES_FILE=$(PWD)/myvalues.yaml
export NUM_NODES=2
export NUM_GPU_PER_NODE=1

DISK_SIZE=10GB
MACHINE_ZONE=europe-west1-b
GCE_PERSISTENT_DISK=gce-nfs-disk
# -----------------------------------------------------------------------------

function run_jobs(){
    . ../kubeutils.sh --source-only
    # Create directory for output
    local OUTPUT=$1
    local USE_CUDA=$2
    mkdir -p ${OUTPUT}

    # Gather the names of workers
    local wnames=($(kube::worker::hostnames))
    local wcount=${#wnames[@]}
    echo "${wcount} workers: ${wnames[@]}"

    ips=($(kube::worker::ips))
    host=$(join_by ',' "${ips[@]}")

    # Copy the file for benchmarking to all workers.
    kube::scp_one ${PWD}/p2p.py /p2p.py

    # make mpirun command
    grid=( 10 100 1000 10000 100000 1000000 10000000 100000000 1000000000 )
    for length in ${grid[@]}
    do
        remote_output="/tmp/${length}"

        cmd="/.openmpi/bin/mpirun -H ${host} /conda/bin/python /p2p.py ${USE_CUDA} --n_iters 100 --length ${length} --foutput ${remote_output}"
        kube::exec_one ${wnames[0]} ${cmd}

        # Download the time from the receiv end.
        kube::download_one ${wnames[1]} ${remote_output} ${OUTPUT}
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

    cpu )
        OUTPUT=output/${arg1}
        run_jobs ${OUTPUT} ""
        ;;

    gpu )
        OUTPUT=output/${arg1}
        run_jobs ${OUTPUT} "--cuda"
        ;;
esac