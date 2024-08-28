#!/bin/bash

set -e
set -u

# Set working directory to the directory of the script
cd "$(dirname "$0")"

source .env
source .env.local
source init-connection.sh

REMOTE_COMMAND="${1:-$REMOTE_COMMAND}"

# Check the existence of used environment variables
TEMP="$HOME $WS_PATH $WS_RESULTS_PATH $USER $HOST $REMOTE_PROJECT_PATH $REMOTE_COMMAND"

# Sync from local to remote
rsync -e "ssh -o 'ControlPath=$HOME/.ssh/ctl/%L-%r@%h:%p'" -avzP --delete --exclude-from=.rsyncignore $WS_PATH/ $USER@$HOST:$REMOTE_PROJECT_PATH/$WS_PATH

# Run command on remote from $REMOTE_PATH/$WS_PATH
ssh -o "ControlPath=$HOME/.ssh/ctl/%L-%r@%h:%p" $USER@$HOST "cd $REMOTE_PROJECT_PATH/$WS_PATH && $REMOTE_COMMAND"

# Sync from remote to local
rsync -e "ssh -o 'ControlPath=$HOME/.ssh/ctl/%L-%r@%h:%p'" -avzP --delete --exclude-from=.rsyncignore $USER@$HOST:$REMOTE_PROJECT_PATH/$WS_RESULTS_PATH/ $WS_RESULTS_PATH