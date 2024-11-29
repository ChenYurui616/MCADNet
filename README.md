# MCAD-Net

This project provides the code and results for 'Efficient Co-salient Object Detection by Integrating Mask Consensus and Attention Diversion'
We provide saliency maps of our and compared methods at [here](https://pan.baidu.com/s/1lakWsUsJb6ePTclrtrL6Ew?pwd=31dm) on three datasets

# Datasets
train dataset:
  gts and images download at here:[coco_duts](https://pan.baidu.com/s/1BYO5WADwvSFoiSYQ1LteFg?pwd=tj3v)

test dataset:
  gts and images download at here:[CoCA,CoSal2015,CoSOD3k](https://pan.baidu.com/s/1Ci0PmHUBGYYWO8UiC2YIiQ?pwd=xqn8)

The structure of the data is as follows:
dataset
├── test
│ ├── gts
│ │ ├── CoCA
│ │ │ ├── Accordion
│ │ │ │ ├── 51499.png
│ │ │ │ ├── 186605.png
│ │ │ ├── .....
│ │ ├── CoSal2015
│ │ │ ├── aeroplane
│ │ │ │ ├── aeroplane_001.png
│ │ │ │ ├── aeroplane_002.png
│ │ │ ├── .....
│ │ ├── CoSOD3k
│ │ │ ├── airplane
│ │ │ │ ├── ILSVRC2012_val_00001390.png
│ │ │ │ ├── ILSVRC2012_val_00004089.png
│ │ │ ├── .....
│ ├── images
│ │ ├── CoCA
│ │ │ ├── .....
│ │ ├── CoSal2015
│ │ │ ├── .....
│ │ ├── CoSOD3k
│ │ │ ├── .....
│ │
├── train
│ ├── coco_duts
│ │ ├── gts
│ │ │ ├── 1
│ │ │ │ ├── ILSVRC2012_test_00006709_syn1.png
│ │ │ │ ├── ILSVRC2012_test_00006709_syn2.png
│ │ │ ├── .....
│ │ ├── images
│ │ │ ├── 1
│ │ │ │ ├── ILSVRC2012_test_00006709_syn1.png
│ │ │ │ ├── ILSVRC2012_test_00006709_syn2.png
│ │ │ ├── .....

- Pretrain weight: [swin224](https://pan.baidu.com/s/1aopiSbXygq5XapcFitVBrA?pwd=3q7c)
- Test datasets: [CoCA,CoSOD3k,CoSal2015](https://github.com/ZhengPeng7/MCCL)
- Train datasets: [coco_duts](https://pan.baidu.com/s/1BYO5WADwvSFoiSYQ1LteFg?pwd=tj3v)
