#!/bin/bash
# Save current git diff, log, status to output directory.
#
# Example usage:
#   ./dump_git_info.sh /path/to/output/directory/
#   ./dump_git_info.sh /path/to/output/directory/ /path/to/git/repo/
#
# Writes the following files to output_dir:
#
# git-status.txt: Output of git status -sb.
# git-log.txt   : Output of
#     git log --graph --pretty='format:%h -%d %s (%cd) <%an>'
# git-diff.txt  : Output of git diff --patch --color=never

function usage {
    echo "Usage: "
    echo "$0 <output_directory> [<git_dir>]"
}

if [[ "$#" != 1 ]] && [[ "$#" != 2 ]] ; then
    echo "Incorrect usage."
    usage
    exit 1
fi

OUTPUT_DIR=$(readlink -f "$1")
if [[ "$#" == 2 ]] ; then
    cd "$2"
fi

git diff --patch --color=never > "${OUTPUT_DIR}/git-diff.patch"
git log --graph --pretty='format:%h -%d %s (%cd) <%an>' \
    > "${OUTPUT_DIR}/git-log.txt"
git status -sb > "${OUTPUT_DIR}/git-status.txt"