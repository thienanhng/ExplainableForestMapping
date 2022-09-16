#!/bin/bash
if [ -z $1 ] 
    then
        echo "Argument missing: directory where dataset should be stored"
    else
        working_dir=$PWD
        echo "The dataset will be stored in $1"
        echo "Downloading and downsampling aerial images..."
        python data/SI_processing/downsample_SI2017.py --url_csv_fn data/csv/download_SI_all.csv --source_dir $1/SwissImage/2017_10cm --dest_dir $1/SwissImage/2017_25cm
        echo
        echo "Downloading the DEM"
        wget -nc -P $1/SwissALTI3D -i data/csv/download_ALTI_all.csv
        cd $1/SwissALTI3D
        rename 's/swissalti3d_2019_/SWISSALTI3D_0.5_TIFF_CHLV95_LN02_/' *.tif
        rename 's/_0.5_2056_5728//' *.tif
        rename 's/-/_/' *.tif
        cd $working_dir
        echo
        echo "Downloading the label data..."
        cd $1
        wget https://zenodo.org/record/7084921/files/TCD_1m.tar.xz?download=1
        tar -xf TCD_1m.tar.xz 
        wget https://zenodo.org/record/7084921/files/TH.tar.xz?download=1 
        tar -xf TH.tar.xz
        wget https://zenodo.org/record/7084921/files/TLMRaster.tar.xz?download=1
        tar -xf TLMRaster.tar.xz
        cd $working_dir
fi
