#!/usr/bin/bash

parallel --keep-order --tag "python3 pulsar_plot.py {} $@" ::: `ls -d plt?????`

echo "Finished plotting"

for vars in $(cat plot_types.txt);
    do
        echo $vars;
        python3 ffmpeg_make_mp4 plt*$vars.png -s -ifps 10 -ofps 30 -name $vars;
    done

echo "Finished making movies"
