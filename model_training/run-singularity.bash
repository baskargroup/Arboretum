#!/bin/bash

args=''
for i in "$@"; do 
  i="${i//\\/\\\\}"
  args="$args \"${i//\"/\\\"}\""
done
echo $args
ls
if [ "$args" == "" ]; then args="/bin/bash"; fi

singularity \
  exec --nv \
  --overlay <singularity_path>.ext3:ro \
  <overlay_path>.sif \
  /bin/bash -c "
 source /ext3/env.sh
 $args 
"