#!/bin/bash

echo "Downloading aerial images"
mkdir -p ${data_dir}/SwissImage/2017_10cm
wget -i -P ${data_dir}/SwissImage/2017_10cm data/csv/SI_download_all.csv
echo "Downsampling the aerial images"
python data/SI_processing/downsample_SI2017.py --source_dir ${data_dir}/SwissImage/2017_10cm --dest_dir ${data_dir}/SwissImage/2017_25cm