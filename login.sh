#!/bin/bash

set -e
set -u

source .env
source .env.local
source init-connection.sh

ssh -o "ControlPath=$HOME/.ssh/ctl/%L-%r@%h:%p" $USER@$HOST