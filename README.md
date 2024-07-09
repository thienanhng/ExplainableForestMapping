# Forest mapping with explainable deep learning

Code for [Nguyen T.-A., Kellenberger B., Tuia D. (2022), *Mapping forest in the Swiss Alps treeline ecotone with explainable deep learning*](https://doi.org/10.1016/j.rse.2022.113217)


```bibtex
@article{Nguyen2022,
title = {Mapping forest in the Swiss Alps treeline ecotone with explainable deep learning},
journal = {Remote Sensing of Environment},
volume = {281},
pages = {113217},
year = {2022},
issn = {0034-4257},
doi = {https://doi.org/10.1016/j.rse.2022.113217},
url = {https://www.sciencedirect.com/science/article/pii/S0034425722003248},
author = {Thiên-Anh Nguyen and Benjamin Kellenberger and Devis Tuia}
```

The goal is to use aerial imagery (SwissImage) and a digital elevation model (DEM, SwissALTI3D) as inputs, and generate a segmentation map with 4 classes: open forest (OF), closed forest (CF), shrub forest (SF), non-forest (NF). 

The baseline model is a *black-box* model (U-net with ResNet-18 encoder) which simply outputs a segmentation map.

<img width="600" alt="baseline_model_flowchart" src="doc/baseline_model_flowchart.png">

The proposed explainable method quantifies two intermediate variables, tree height (TH) and tree canopy density (TCD), and combines these intermediate predictions using logical rules derived from the class definitions (*rule module*), to obtain a rule-based segmentation map. The model also outputs *correction activations* which are combined with the rule-based predictions (*correction module*) to obtain the final predictions. 

<img width="600" alt="sb_model_flowchart" src="doc/sb_model_flowchart.png">

## Data information/download (in construction):

The lists of tiles used in the training, validation and test sets are specified in the csv files in folder [data/csv](data/csv) (naming convention <source_1>\_<source_2>\_...\_\<set>.csv). The training, validation and test set spatial repartition is the following:

<img width="600" alt="aoi_w_splits_w_legend" src="doc/aoi_w_splits_w_legend.png">

- input data:
  - [SwissImage data](https://www.swisstopo.admin.ch/en/geodata/images/ortho/swissimage10.html#download)
  - [SwissALTI3D data](https://www.swisstopo.admin.ch/en/geodata/height/alti3d.html#download)
- target data:
  - original data:
    - [SwissTLM3D vector data](https://www.swisstopo.admin.ch/en/geodata/landscape/tlm3d.html#download)
    - [VHM NFI data](https://www.envidat.ch/dataset/vegetation-height-model-nfi)
  - processed data:
    - rasterized forest targets obtained from SwissTLM3D and TH et TCD targets obtained by processing the VHM. For download, see [here](data/get_dataset.sh)

Geo-located plot data from Swiss National Forest Inventory (NFI) that we used for comparison is not currently openly available. More info [here](https://lfi.ch/lfi/lfi-en.php).

## Getting started


### Python Dependencies

Anaconda environment
```
Coming soon
```
### Download dataset (in construction)

Run the following command to download the full dataset (aerial imagery, DEM, rasterized and processed targets). Replace `../Data` with the folder where you want the dataset to be stored. Make sure you have 90 GB (!) of free space.
```bash
. data/get_dataset.sh ../Data
```
### Training and inference (in construction)

The python scripts to train a model or perform inference are [train.py](train.py) and [infer.py](infer.py) respectively. Bash scripts corresponding to each experiment in the paper are available [here](launch_scripts/), with naming convention *launch\_\<task>\_<experiment\_name>.sh*. You can edit these scripts to run your own training/inference process.

The experiments are:
  - bb: black-box model (BB), which is the baseline model
  - sb: semantic bottleneck model (SB), which is the proposed explainable model
  - section 5.2.4 experiments:
    - sb_corrp: semantic bottleneck model trained with a looser correction penality (SBcorr<sup>+</sup>)
    - sb_rulem: semantic bottleneck model trained with a looser rule enforcement (SBrule<sup>-</sup>)
  - appendix experiments:
    - bb_alti_ablation: black-box model trained without using the DEM as input
    - bb_flat: black-box model trained for a single, non-hierarchical segmentation task
  
For each experiment listed above, the trained model weights as well as the metrics at each epoch are provided; when needed (for inference or fine-tuning), they are automatically downloaded. All the results can be visualized in the notebooks in the [notebooks](notebooks/) folder.
  
## Warnings
- make sure to update the filepaths in the csv files in [data/csv](data/csv) according to where you stored the data, and which tiles you are using.
- the data preprocessing constants (mean and standard deviation values) of the input data are hardcoded [in this script](utils/ExpUtils.py) and were computed on the training set using [this script](data/get_statistics.py). They should be modified if using a different training set.
