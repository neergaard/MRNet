#!/bin/bash

wget http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/data/metadata.csv

for i in '01' '02' '03' '04' '05' '06' '07' '08' '09' '10'

    do
        wget http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/data/volumetric_data/vol$i.7z
        mkdir vol$i; mv vol$i.7z vol$i
        cd vol$i
        7za e vol$i.7z; rm vol$i.7z
        cd ..
    done

