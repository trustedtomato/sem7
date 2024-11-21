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

OPT_GROUP=${GROUP:+'--groupmap=*:'"$GROUP"}

OPT_RSYNCIGNORE_LOCAL=""
if [ -f .rsyncignore_local ]; then
  OPT_RSYNCIGNORE_LOCAL="--exclude-from=.rsyncignore_local"
fi

# Sync from local to remote
rsync -e "ssh -o 'ControlPath=$HOME/.ssh/ctl/%L-%r@%h:%p'" -urltvzP --perms --chmod=770 $OPT_GROUP $OPT_RSYNCIGNORE_LOCAL --exclude-from=.rsyncignore $WS_PATH/ $USER@$HOST:$REMOTE_PROJECT_PATH/$WS_PATH

# Run command on remote from $REMOTE_PATH/$WS_PATH
ssh -o "ControlPath=$HOME/.ssh/ctl/%L-%r@%h:%p" $USER@$HOST "cd $REMOTE_PROJECT_PATH/$WS_PATH && source ./.venv/bin/activate && python ts2vec_run_experiments.py"

# Sync from remote to local
rsync -e "ssh -o 'ControlPath=$HOME/.ssh/ctl/%L-%r@%h:%p'" -urltvzP $OPT_RSYNCIGNORE_LOCAL --exclude-from=.rsyncignore $USER@$HOST:$REMOTE_PROJECT_PATH/$WS_PATH/ $WS_PATH