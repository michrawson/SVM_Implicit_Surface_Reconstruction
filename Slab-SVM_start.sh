#!/bin/bash

trap ctrl_c INT

function ctrl_c() {
    is_killed=1
}

is_killed=0

for file_name in `ls *.ply | tr '\n' '\n'`
do
    for v in .9999 .999 .99 .99 .9 .8 .7 .6 .5 .4 .3 .2 .1 .01
    do
        for sigma in 5 10 20 50 60 70 80 90 100 110 120 130 200 500
        do
            if [ $sigma -eq 500 ]
            then
                if [ $is_killed -eq 0 ]
                then
                    python Slab-SVM.py $file_name $v $sigma
                fi
                jobs
            else
                if [ $is_killed -eq 0 ]
                then
                    python Slab-SVM.py $file_name $v $sigma &
                fi
            fi
        done
    done
done
