# MRNet

Dataset from Clinical Hospital Centre Rijeka, Croatia, originally appears in:

I. Štajduhar, M. Mamula, D. Miletić, G. Unal, Semi-automated detection of anterior cruciate ligament injury from MRI, Computer Methods and Programs in Biomedicine, Volume 140, 2017, Pages 151–164. (http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/data/Stajduhar2017.pdf)

## Setup

`bash download.sh` (caution: downloads ~6.68 GB of data)

`conda env create -f environment.yml`

`source activate mrnet`

## Train

`python train.py --rundir [experiment name] --diagnosis 0 --gpu`

- diagnosis is highest diagnosis allowed for negative label (0 = injury task, 1 = tear task)
- arguments saved at `[experiment-name]/args.json`
- prints training & validation metrics (loss & AUC) after each epoch
- models saved at `[experiment-name]/[val_loss]_[train_loss]_epoch[epoch_num]`

## Evaluate

`python evaluate.py --split [train/valid/test] --diagnosis 0 --model_path [experiment-name]/[val_loss]_[train_loss]_epoch[epoch_num] --gpu`

- prints loss & AUC
