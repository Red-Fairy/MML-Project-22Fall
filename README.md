## Towards improvement to CLIPStyler (PKU Multi Modal Learning Course Project)
### Authors: Rundong Luo, Zekai Zhang, Chuanruo Ning (Equal Contributions)

**Our project is based on the [CLIPStyler (CVPR 2022)](https://github.com/cyclomon/CLIPstyler)**

[**Presentation Slides**](./Project-Presentation.pdf), [**Project Report**](./Project-Report.pdf)

![MAIN3_e2-min](https://user-images.githubusercontent.com/94511035/142139437-9d91f39e-b3d7-46cf-b43b-cb7fdead69a8.png)


### Environment
Pytorch 1.7.1, Python 3.6

```
$ conda create -n CLIPstyler python=3.6
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ conda install -c anaconda git
$ pip install git+https://github.com/openai/CLIP.git
```

## Original Style Transfer (in paper Clipstyler)
### Style Transfer with Single-image
We provide demo with replicate.ai 
<a href="https://replicate.ai/paper11667/clipstyler"><img src="https://img.shields.io/static/v1?label=Replicate&message=Demo and Docker Image&color=blue"></a>

To train the model and obtain the image, run

```
python train_CLIPstyler.py --content_path ./test_set/face.jpg \
--content_name face --exp_name exp1 \
--text "Sketch with black pencil"
```

To change the style of custom image, please change the ```--content_path``` argument

edit the text condition with ```--text``` argument

For easy demo, we provide Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dg8PXi-TVtzdpbaoI7ty72SSY7xdBgwo?usp=sharing).

*Warning : Due to slow computation speed of colab, it may take several minutes in colab environment

### Fast Style Transfer
Before training, plase download DIV2K dataset [LINK](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

We recommend to use Training data of High-Resolution(HR) images.

To train the model, please download the pre-trained vgg encoder & decoder models in [LINK](https://drive.google.com/drive/folders/17UDzXtp9IZlerFjGly3QEm2uU3yi7siO?usp=sharing).

Please save the downloaded models in ```./models``` directory

Then, run the command

```
python train_fast.py --content_dir $DIV2K_DIR$ \
--name exp1 \
--text "Sketch with black pencil" --test_dir ./test_set
```

Please set the ```$DIV2K_DIR$``` as the directory in which DIV2K images are saved.

To test the fast style transfer model, 

```
python test_fast.py --test_dir ./test_set --decoder ./model_fast/clip_decoder_iter_200.pth.tar
```

Change the argument ```--decoder``` to other trained models for testing on different text conditions.

We provide several fine-tuned decoders for several text conditions. [LINK](https://drive.google.com/drive/folders/1U-4tEigPaJxfXRMnEdRDtyQ99O5ondrs?usp=sharing)

To use high-resolution image, please add ```--hr_dir ./hr_set``` to test command. 

We provide colab notebook for testing fast transfer model [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sdvRuBECA48sPPlBb7UTOuk9peGggdI9?usp=sharing)

### Style interpolation on Fast Style Transfer

Style interpolation results with interpolating weight parameters of two fine-tuned decoder models


To interpolate the fast style transfer model, 

```
python test_intp.py --decoder_src $SOURCE_DECODER --decoder_trg $TARGET_DECODER
```

Put source and target decoder model paths in ```$SOURCE_DECODER``` and ```$TARGET_DECODER```

Style interpolation example with interpolating two styles "Stone wall" and "Desert sand"

### Video style transfer with Fast model

For video style transfer, first install video io package
```
$ pip install imageio-ffmpeg
$ conda install -c conda-forge/label/cf202003 opencv
```

Then run the following command,

```
python test_video.py --content_path $VIDEO_PATH$ --decoder $DECODER_PATH$
```

## Improved CLIPstyler (our improvements)
We realize three improvements for the CLIPstyler. The methods and experiments could be
find in the report.
### Multicrop CLIPstyler
WE exploit the patch-wise loss between each content patch and stylized patch.

run the following command,

```
python 
```
### Localized CLIPstyler
We realize localized style transfer. You can assign the specific part of the image that you 
want to stylize.

To see the results, run
```
python train_CLIPstyler_localize.py --content_path $imageyou want to stylize$ \
--content_name face --exp_name exp1 \
--source_text $the content you want to stylize$ \
--text $the style you want to apply$
```
for example
```
python train_CLIPstyler_localize.py --content_path ./test_set/boat.jpg \
--content_name boat --exp_name exp1 \
--source_text water \
--text stars
```
this code will transfer the "water" in image "boat.jpg" into "stars" style
### Style-aware Selective Sampling
This improvement allows control over the extent of stylization.

run the following command to see the results,

```
python 
```
