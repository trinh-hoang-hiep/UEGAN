##  UEGAN


## Setup
 * Install anaconda: https://www.anaconda.com/distribution/
 * set up conda environmet w/ python 3.8, ex: `conda create --name uegan python=3.8`
 * `conda activate uegan`
 * `pip install -r requirements.txt`
 
## Datasets
 * Create a folder for Datasets
 * **CAMO**
 * **Polyp**

## Running

```bash
python train.py --task=COD  --uncer_method ganabp --ckpt {checkpoint_file}
!python  test.py  --uncer_method ganabp  --task COD --ckpt {new_checkpoint_file}
```

## Results
Results will be saved in a folder named `experiments/`. To get the final average accuracy, retrieve the final number in the file `experiments/**/save_images/**/results_*_epoch.csv`

## Results Download
Results of our SINet can be found in  [download link](https://drive.google.com/drive/folders/12V2h7ujNuky4AWwWGEKc7eXt2e4JentP?usp=sharing), you can manually download it 

## Evaluation 
Download it in [here](https://github.com/DengPingFan/PraNet/tree/master/eval) and [here](https://github.com/trinh-hoang-hiep/UEGAN/tree/main/uncertainty_metric)