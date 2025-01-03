# MCADNet
This project provides the code and results for 'Efficient Co-salient Object Detection by Integrating Mask Consensus and Attention Diversion'.
We provide saliency maps of our method at [MCADNet_Salmap](https://pan.baidu.com/s/1lakWsUsJb6ePTclrtrL6Ew?pwd=31dm) on three datasets.

# Getting Started
- Environment configuration:<br>
pytorch==1.13.1;cud==11.7.0;python==3.8<br>
pip install numpy==1.17.4 torch==1.9.1 torchvision==0.10.1 einops opencv-python timm   tqdm


# Datasets
- train dataset: 
  gts and images download at here:[coco_duts](https://pan.baidu.com/s/1BYO5WADwvSFoiSYQ1LteFg?pwd=tj3v), (Please be careful to unpack both 'images_1' and 'images_2' into the same folder 'images','gts' can be unpacked directly into the 'coco_duts' folder).

- test dataset: 
  gts and images download at here:[CoCA,CoSal2015,CoSOD3k](https://pan.baidu.com/s/1Ci0PmHUBGYYWO8UiC2YIiQ?pwd=xqn8), ('images'and'gts' can be unpacked directly into the 'test' folder).

The structure of the data is as follows:

├ dataset<br>
├── test<br>
│ ├── gts<br>
│ │ ├── CoCA<br>
│ │ │ ├── Accordion<br>
│ │ │ │ ├── 51499.png<br>
│ │ │ │ ├── 186605.png<br>
│ │ │ ├── .....<br>
│ │ ├── CoSal2015<br>
│ │ │ ├── aeroplane<br>
│ │ │ │ ├── aeroplane_001.png<br>
│ │ │ │ ├── aeroplane_002.png<br>
│ │ │ ├── .....<br>
│ │ ├── CoSOD3k<br>
│ │ │ ├── airplane<br>
│ │ │ │ ├── ILSVRC2012_val_00001390.png<br>
│ │ │ │ ├── ILSVRC2012_val_00004089.png<br>
│ │ │ ├── .....<br>
│ ├── images<br>
│ │ ├── CoCA<br>
│ │ │ ├── .....<br>
│ │ ├── CoSal2015<br>
│ │ │ ├── .....<br>
│ │ ├── CoSOD3k<br>
│ │ │ ├── .....<br>
│ │<br>
├── train<br>
│ ├── coco_duts<br>
│ │ ├── gts<br>
│ │ │ ├── 1<br>
│ │ │ │ ├── ILSVRC2012_test_00006709_syn1.png<br>
│ │ │ │ ├── ILSVRC2012_test_00006709_syn2.png<br>
│ │ │ ├── .....<br>
│ │ ├── images<br>
│ │ │ ├── 1<br>
│ │ │ │ ├── ILSVRC2012_test_00006709_syn1.png<br>
│ │ │ │ ├── ILSVRC2012_test_00006709_syn2.png<br>
│ │ │ ├── .....<br>

# Directory
- Root<br>
├ evaluation<br>
├ evaluator<br>
├ dataset<br>
├ model<br>
├ predicton<br>
├ pth<br>
├ weight<br>

# Training
Download swin224 at [swin224](https://pan.baidu.com/s/1aopiSbXygq5XapcFitVBrA?pwd=3q7c), and put it in "./pth/", 
modify paths of data, then run ```python train.py```.
# Testing
After training, the weights are stored in the 'weight' folder, then change the quotes in lines 22 and 26 of the test file to <br>```modelname = os.path.join(args.param_root, 'epochname')
``` and <br>```yynet_dict = torch.load(os.path.join(args.param_root, 'epochname'))```,<br>where 'epochname' refers to the best weight name in training.<br>
then run ```python test.py```.<br>
The pre-trained weights for our method at [MCADNet_Pre](https://pan.baidu.com/s/1FCVAI9ixSk3f80i6D5FXRQ?pwd=ea2v). If you want to use our trained model, download and put it in "./weight/", then change the quotes in lines 22 and 26 of the test file to <br>```modelname = os.path.join(args.param_root, '93.pt')
``` and <br>```yynet_dict = torch.load(os.path.join(args.param_root, '93.pt'))```,then run ```python test.py```.<br>
# Evaluating
After testing, the prediction maps are stored in the 'prediction' folder,then run ```python evaluate.py```.
