#!/bin/bash

PLATFORM_PYTHON_VERSION="Python 3.7.5"

if [[ $(python --version) != $PLATFORM_PYTHON_VERSION ]] ; then
    echo "PLATFORM_PYTHON_VERSION is set to $PLATFORM_PYTHON_VERSION (change if it's not correct), use same python verson to make wheels (current is $(python --version))"
    exit 0
fi

# ensure wheel is installed
pip install wheel


output_dir=$(pwd)/platform_archives
mkdir -p $output_dir

current_dir=$(pwd)

wheelhouse_tmp_dir=$(mktemp -d -t ci-XXXXXXXXXX)


# build wheels from requirements
pip wheel -r requirements.txt -w $wheelhouse_tmp_dir

# archive to wheelhouse.tar
wheelhouse_tar_name=wheelhouse-$(date +"%FT%H%M").tar
cd $wheelhouse_tmp_dir
tar -cvf $wheelhouse_tar_name  *
cp $wheelhouse_tar_name $output_dir
cd $current_dir
# remove wheelhouse dir
rm -rf $wheelhouse_tmp_dir