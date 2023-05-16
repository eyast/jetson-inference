#!/usr/bin/env bash

source docker/containers/scripts/l4t_version.sh

CONTAINER_IMAGE="jetson-inference:r$L4T_VERSION"

if [ $L4T_RELEASE -eq 32 ]; then
    if [[ $L4T_REVISION_MAJOR -lt 4 && $L4T_REVISION_MINOR -gt 4 ]]; then
        # L4T R32.4 was the first version containers are supported on
        version_error
    elif [ $L4T_REVISION_MAJOR -eq 5 ]; then
        # L4T R32.5.x all run the R32.5.0 container
        CONTAINER_IMAGE="jetson-inference:r32.5.0"
    elif [ $L4T_REVISION_MAJOR -eq 7 ]; then
        # L4T R32.7.x all run the R32.7.0 container
        CONTAINER_IMAGE="jetson-inference:r32.7.1"
    fi
elif [ $L4T_RELEASE -eq 35 ]; then
    if [ $L4T_REVISION_MAJOR -gt 3 ]; then
        CONTAINER_IMAGE="jetson-inference:r35.3.1"
    fi
fi

CONTAINER_REMOTE_IMAGE="dustynv/$CONTAINER_IMAGE"

TAG=$CONTAINER_IMAGE

if [[ "$(sudo docker images -q $CONTAINER_IMAGE 2> /dev/null)" == "" ]]; then
	CONTAINER_IMAGE=$CONTAINER_REMOTE_IMAGE
fi