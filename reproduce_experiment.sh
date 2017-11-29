#!/bin/bash

set -e

function usage() {
    echo "Usage: $0 <experiment_dir> <output_root>"
}

if [[ "$#" < 2 ]] ; then
    usage >&2
    exit 1
fi

experiment_dir="$1"
output_root="$2"

if [[ ! -d ${experiment_dir} ]] ; then
    echo "ERROR: Experiment directory does not exist!" >&2
    exit 1
fi

if [[ ! -d ${output_root} ]] ; then
    echo "ERROR: Output root directory does not exist!" >&2
    exit 1
fi

experiment_id=$(cat ${experiment_dir}/experiment-id.txt)
if [[ -e "${experiment_id}/git-head.txt" ]] ; then
    commit_id=$(cat ${experiment_dir}/git-head.txt)
else
    short_commit_id=$(\
        head -1 ${experiment_dir}/git-log.txt \
        | sed -e 's/^\* \([^ ]*\).*/\1/g')
    commit_id=$(git rev-parse ${short_commit_id})
fi

git worktree add --detach $(readlink -e ${output_root})/${experiment_id}
echo "### Created worktree at ${output_root}/${experiment_id}"

cd ${output_root}/${experiment_id}

git checkout ${commit_id} > /dev/null
echo "### Checked out ${commit_id}"

git apply ${experiment_dir}/git-diff.patch
echo "### Applied git diff patch"

ln -s ${experiment_dir}/config.yaml config-${experiment_id}.yaml
echo "### Copied config to config-${experiment_id}.yaml"

echo "### Compiling video_util"
git submodule update --init --recursive
cd video_util
make
# Old versions of code accidentally imported video_frames_pb.lua from the
# video_util directory instead of the util subdirectory.
ln -s util/video_frames_pb.lua video_frames_pb.lua
