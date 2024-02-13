HERE=$(pwd) # Absolute path of current directory

user=`whoami`
uid=`id -u`
gid=`id -g`

#echo "$user $uid $gid"

DOCKER_REPO="dimdano/"

BRAND="transaxx"
VERSION="1.0"

IMAGE_NAME=${DOCKER_REPO}$BRAND:${VERSION}

docker run \
    -e USER=$user -e UID=$uid -e GID=$gid \
    -v $HERE:/workspace/ \
    -w /workspace/ \
    -it \
    --gpus '"device=0"' \
    --rm \
    --network=host \
    $IMAGE_NAME \
    /bin/sh -c "/workspace/../etc/banner.sh; bash"
