# SFA-Net

- This repository presents an architecture for urban scene segmentation in high-resolution remote sensing images, with support for both training and testing.

## Folder Structure

Prepare the following folders to organize this repo:
```none
SFA-Net
├── network
├── config
├── tools
├── model_weights (save the model weights)
├── fig_results (save the masks predicted)
├── lightning_logs (CSV format training logs)
├── data
│   ├── LoveDA
│   │   ├── Train
│   │   │   ├── Urban
│   │   │   │   ├── images_png (original images)
│   │   │   │   ├── masks_png (original masks)
│   │   │   │   ├── masks_png_convert (converted masks used for training)
│   │   │   │   ├── masks_png_convert_rgb (original rgb format masks)
│   │   │   ├── Rural
│   │   │   │   ├── images_png 
│   │   │   │   ├── masks_png 
│   │   │   │   ├── masks_png_convert
│   │   │   │   ├── masks_png_convert_rgb
│   │   ├── Val (the same with Train)
│   │   ├── Test
│   │   ├── train_val (Merge Train and Val)
│   ├── uavid
│   │   ├── uavid_train (original)
│   │   ├── uavid_val (original)
│   │   ├── uavid_test (original)
│   │   ├── uavid_train_val (Merge uavid_train and uavid_val)
│   │   ├── train (processed)
│   │   ├── val (processed)
│   │   ├── train_val (processed)
│   ├── vaihingen
│   │   ├── train_images (original)
│   │   ├── train_masks (original)
│   │   ├── test_images (original)
│   │   ├── test_masks (original)
│   │   ├── test_masks_eroded (original)
│   │   ├── train (processed)
│   │   ├── test (processed)
│   ├── potsdam (the same with vaihingen)
```


## Data Preprocessing

Download Datasets
- [ISPRS Vaihingen, Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)
- [UAVid](https://uavid.nl/)
- [LoveDA](https://codalab.lisn.upsaclay.fr/competitions/421)

Configure the folder as shown in 'Folder Structure' above.

## Acknowledgement

- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [GeoSeg](https://github.com/WangLibo1995/GeoSeg)

# Notes
- Since nn.CrossEntropyLoss requires that each useful pixel label should begain from 0, all the label should be `-1`
    - so in original LoveDA data, there are 6 classes, the label are from 0-7, new label is still 0-7
    - background : 1 --> 0
    - building : 2 --> 1
    - road : 3 --> 2
    - water : 4 --> 3
    - barren : 5 --> 4
    - forest : 6 --> 5
    - agricultural : 7 --> 6
- and original 0 should be n

