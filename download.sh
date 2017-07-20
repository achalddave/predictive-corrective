#!/bin/bash

wget https://cmu.box.com/shared/static/xucg15tye9cr03da1llbxwufsm8mumh6.gz -O data.tar.gz
mkdir -p data
echo "Outputting data to data/"
# Tar was uploaded with upload/ as the main directory; change to data/
tar xzvf data.tar.gz --strip-components 1 -C data | sed -e 's/^upload/data/g'

# Remove README file that is not useful for end users.
rm data/multithumos/models/README.md
