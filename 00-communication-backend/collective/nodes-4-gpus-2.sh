#!/bin/bash

# ---------------------------------------
# Run all reduce on 8 nodes with 1 GPU.
# ---------------------------------------

# ---------------------------- Cluster Configs -------------------------------`
export CLUSTER_NAME=nodes-4-gpus-2
export RELEASE_NAME=rel
export MYVALUES_FILE=$(PWD)/myvalues.yaml
export NUM_NODES=4
export NUM_GPU_PER_NODE=2
export MACHINE_TYPE=n1-standard-4

DISK_SIZE=10GB
MACHINE_ZONE=europe-west1-b
GCE_PERSISTENT_DISK=gce-nfs-disk
# -----------------------------------------------------------------------------

function run_mpi_jobs(){
    . ../kubeutils.sh --source-only
    # Create directory for output
    local OUTPUT=$1
    mkdir -p ${OUTPUT}

    # Gather the names of workers
    local wnames=($(kube::worker::hostnames))
    local wcount=${#wnames[@]}
    echo "${wcount} workers: ${wnames[@]}"

    ips=($(kube::worker::ips))
    host=$(join_by ',' "${ips[@]}")

    # Copy the file for benchmarking to all workers.
    kube::scp_one ${PWD}/mpi.py /mpi.py

    # make mpirun command
    grid=( 10 100 1000 10000 100000 1000000 10000000 100000000 1000000000 )
    for length in ${grid[@]}
    do
        remote_output="/tmp/${length}"

        cmd="/.openmpi/bin/mpirun -H ${host} --mca btl_openib_want_cuda_gdr 1 /conda/bin/python /mpi.py --n_iters 100 --length ${length} --foutput ${remote_output}"
        kube::exec_one ${wnames[0]} ${cmd}

        # Download the time from the receiv end.
        kube::download_one ${wnames[0]} ${remote_output} ${OUTPUT}
    done
}

function run_nccl_jobs(){
    . ../kubeutils.sh --source-only
    # Create directory for output
    local OUTPUT=$1
    mkdir -p ${OUTPUT}

    # Gather the names of workers
    local wnames=($(kube::worker::hostnames))
    local wcount=${#wnames[@]}
    echo "${wcount} workers: ${wnames[@]}"

    ips=($(kube::worker::ips))
    host=$(join_by ',' "${ips[@]}")

    # Copy the file for benchmarking to all workers.
    kube::scp_one ${PWD}/nccl.py /nccl.py

    # make mpirun command
    grid=( 10 100 1000 10000 100000 1000000 10000000 100000000 1000000000 )
    for length in ${grid[@]}
    do
        remote_output="/tmp/${length}"
        WORLD_SIZE=${wcount}

        for RANK in $(seq 0 $(( $WORLD_SIZE - 1 )) )
        do
            MASTER_PORT=1234
            MASTER_ADDR=${ips[0]}
            cmd="/conda/bin/python -m torch.distributed.launch --nproc_per_node=1 --nnodes=${NUM_NODES} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} /nccl.py --n_iters 100 --length ${length} --foutput ${remote_output}"
            kube::exec_one ${wnames[${RANK}]} ${cmd} &
            pids[${i}]=$!
            echo ${pids[${i}]}
        done

        for pid in ${pids[*]}; do
            wait $pid
        done

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

    reinstall )
        bash ../../launch.sh uninstall
        bash ../../launch.sh install

        . ../kubeutils.sh --source-only

        # Fix the symbolic link problem
        # TODO: Solve this problem in docker image
        kube::scp_one ${PWD}/fix_link.sh /fix_link.sh
        kube::exec_all "bash /fix_link.sh"
        ;;

    mpi )
        OUTPUT=output/${CLUSTER_NAME}/${arg1}
        run_mpi_jobs ${OUTPUT}
        ;;

    nccl )
        OUTPUT=output/${CLUSTER_NAME}/${arg1}
        run_nccl_jobs ${OUTPUT}
        ;;
esac