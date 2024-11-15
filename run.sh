#!/bin/bash

set -e
set -u

# Set working directory to the directory of the script
cd "$(dirname "$0")"

source .env
source .env.local
source init-connection.sh

# Check the existence of used environment variables
TEMP="$HOME $WS_PATH $USER $HOST $REMOTE_PROJECT_PATH $REMOTE_COMMAND"

# Make remote target directory
ssh -o "ControlPath=$HOME/.ssh/ctl/%L-%r@%h:%p" $USER@$HOST mkdir -p $REMOTE_PROJECT_PATH/$WS_PATH

# set groupmap to --groupmap='*:'"$GROUP" if GROUP is set, otherwise set it to empty string
GROUP=${GROUP:+'--groupmap=*:'"$GROUP"}

# Sync from local to remote
rsync -e "ssh -o 'ControlPath=$HOME/.ssh/ctl/%L-%r@%h:%p'" -urltvzP --perms --chmod=770 $GROUP --exclude-from=.rsyncignore $WS_PATH/ $USER@$HOST:$REMOTE_PROJECT_PATH/$WS_PATH

# Run command on remote from $REMOTE_PATH/$WS_PATH
ssh -o "ControlPath=$HOME/.ssh/ctl/%L-%r@%h:%p" $USER@$HOST "cd $REMOTE_PROJECT_PATH/$WS_PATH && $REMOTE_COMMAND $@"

# Sync from remote to local
rsync -e "ssh -o 'ControlPath=$HOME/.ssh/ctl/%L-%r@%h:%p'" -urltvzP --exclude-from=.rsyncignore $USER@$HOST:$REMOTE_PROJECT_PATH/$WS_PATH/ $WS_PATH