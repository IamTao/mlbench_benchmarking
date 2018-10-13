#!/bin/bash

function kube::worker::hostnames(){
    # Get a sequence of names; if you want an array, use additional parentheses ($(kube::worker::hostnames))
    kubectl get pods | grep 'worker' | awk '{print $1}'
}

function kube::worker::ips(){
    # Get a sequence of names; if you want an array, use additional parentheses ($(kube::worker::hostnames)) 
    kubectl get pods -o wide | grep worker | awk '{print $6}'
}

function join_by(){ local IFS="$1"; shift; echo "$*"; }

function kube::exec_one(){
    # Execute command on one of nodes
    local input=($@)
    local node=${input[0]}
    local command=${input[@]:1}
    echo "kubectl exec $node -- $command"
    kubectl exec ${node} -- $command
}

# function kube::scp_all(){
#     local namespace=default
#     local names=$(kube::worker::hostnames)
#     for node in ${names[@]}
#     do
#         echo "kubectl cp ${PWD}/src ${namespace}/${node}:/src"
#         kubectl cp ${PWD}/src ${namespace}/${node}:/
#     done
# }

function kube::scp_one(){
    # COPY one file to all of the nodes
    local namespace=default
    local names=$(kube::worker::hostnames)
    local from=$1
    local to=$2
    for node in ${names[@]}
    do
        echo "kubectl cp ${from} ${namespace}/${node}:${to}"
        kubectl cp ${from} ${namespace}/${node}:${to}
    done
}

function kube::download_one(){
    local namespace=default
    local node=$1
    local path=$2
    local output=$3
    kubectl cp ${namespace}/${node}:${path} ${output}
}

function kube::exec_all(){
    # Execute command on one of nodes
    local command=$1
    local names=$(kube::worker::hostnames)
    for node in ${names[@]}
    do
        echo "kubectl exec $node -- $command"
        kubectl exec $node -- $command
    done
}