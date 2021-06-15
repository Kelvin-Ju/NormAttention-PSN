# NormAttention-PSN

## NormAttention-PSN: A High-frequency Region Enhanced Photometric Stereo Network with Normalized Attention. (Submitting to IJCV)

Yakun JU, Boxin SHI, Muwei JIAN, Junyu DONG, and Kin-Man LAM


## Environment

Implemented in PyTorch with Ubuntu 18.04.

Python: 3.6.9 

PyTorch 1.40 with scipy, numpy, etc.

CUDA 10.0

RTX 2080 (8G)

## For training our NormAttention-PSN, you need download these two datasets:
Blobby shape dataset (4.7 GB), and Sculpture shape dataset (19 GB), via: 

```shell
sh scripts/download_synthetic_datasets.sh
```
## For testing our NormaAttention-PSN, you can download these datsets:

DiLiGenT main dataset (default) (850MB), via:
```shell
sh scripts/prepare_diligent_dataset.sh  
```
or   https://drive.google.com/file/d/1EgC3x8daOWL4uQmc6c4nXVe4mdAMJVfg/view

DiLiGenT test dataset (759MB), via:

https://drive.google.com/file/d/1LzRMwrxWMdV_ASYzUMm9ZlAmyBs-QJRs/view

Light Stage Data Gallery, via:

https://vgl.ict.usc.edu/Data/LightStage/

(We advise to down-sample the resolution of this dataset, otherwise, your GPU is really hard to handle.)

## Testing on your device:
```shell
python eval/run_model.py --retrain data/models/NormAttention-PSN-test.pth.tar --in_img_num X --normalize --train_img_num 32
```
You can change X to adjust the number of the input image. 

NormAttention-PSN-test.pth.tar is our pre-trained weights (same as the report in paper).

(up to 96 in the DiLiGenT main/test dataset, and 253 in the Light Stage Data Gallery)

## Training on your device:
```shell
python main.py --concat_data --in_img_num 32 --normalize --item normalize
```
Defualt: training with 32 input images.


## Acknowledgement:

Our code is partially based on: https://github.com/guanyingc/PS-FCN.

## Our Previous work:

Attention-PSN, published on IJCAI 2020. https://www.ijcai.org/Proceedings/2020/0097 





