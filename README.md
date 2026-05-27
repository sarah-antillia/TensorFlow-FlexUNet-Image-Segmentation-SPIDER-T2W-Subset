<h2>TensorFlow-FlexUNet-Image-Segmentation-SPIDER-T2W-Subset (2026/05/28)</h2>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>SPIDER: Lumber Spine MRI Subset </b> based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>), and a PNG
 <a href="https://drive.google.com/file/d/1MabgzvEJy2NBbKzWYFVMfo3t2WsWAewo/view?usp=sharing">
SPIDER-T2W-ImageMask-Subset.zip</a> with colorized masks (<a href="https://creativecommons.org/licenses/by/4.0/legalcode">
Creative Commons Attribution 4.0 International
</a>), which was derived by us from <br><br>
<a href="https://zenodo.org/records/10159290">
<b>SPIDER - Lumbar spine segmentation in MR images: a dataset and a public benchmark</b>
</a>
<br><br>
<hr>
<b>Actual Image Segmentation for SPIDER-T2W-Subset Images </b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks.
<br><br>
<a href="#color-class-mapping-table">Class color mapping table</a>
<br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test/images/1007_5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test/masks/1007_5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_output/1007_5.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test/images/1140_10.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test/masks/1140_10.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_output/1140_10.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test/images/1176_5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test/masks/1176_5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_output/1176_5.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The dataset used here was derived from <br><br>
<b>images.zip</b> and <b>masks.zip</b> in 
<a href="https://zenodo.org/records/10159290">
<b>SPIDER - Lumbar spine segmentation in MR images: a dataset and a public benchmark</b>
</a> on the zenodo.org.
<br><br>

The following explanation was taken from the above zenodo site. 
<br><br>
This is a large publicly available multi-center lumbar spine magnetic resonance imaging (MRI) dataset with reference segmentations 
of vertebrae, intervertebral discs (IVDs), and spinal canal. <br>
The dataset includes 447 sagittal T1 and T2 MRI series from 218 studies of 218 patients with a history of low back pain. 
The data was collected from four different hospitals. <br>
There is an additional hidden test set, not available here, used in the accompanying SPIDER challenge on spider.grand-challenge.org. <br>
We share this data to encourage wider participation and collaboration in the field of spine segmentation, 
and ultimately improve the diagnostic value of lumbar spine MRI.
<br><br>
Which MRI studies are assigned to the training and validation sets can be found in the overview file. 
This file also provides the biological sex for all patients and the age for the patients for which 
this was available. It also includes a number of scanner and acquisition parameters for each individual MRI study. <br>
The dataset also comes with radiological gradings found in a separate file for the following degenerative changes:
<br><br>
1.    Modic changes (type I, II or III)<br>
2.    Upper and lower endplate changes / Schmorl nodes (binary)<br>
3.    Spondylolisthesis (binary)<br>
4.    Disc herniation (binary)<br>
5.    Disc narrowing (binary)<br>
6.    Disc bulging (binary)<br>
7.    Pfirrman grade (grade 1 to 5)<br>. 
<br>
<b>Citation</b><br>
Jasper Willem van der Graaf, Miranda L. van Hooff, Constantinus F. <br>
M. Buckens, Matthieu Rutten, Job L. C. van Susante, Robert Jan Kroeze, <br>
Marinus de Kleuver, Bram van Ginneken, & Nikolas Lessmann. <br>
(2023). SPIDER - Lumbar spine segmentation in MR images: a dataset and a public benchmark [Data set]. Zenodo.<br>
<a href="https://doi.org/10.5281/zenodo.10159290">https://doi.org/10.5281/zenodo.10159290</a>
<br><br>
<b>License</b><br>
<a href="https://creativecommons.org/licenses/by/4.0/legalcode">
Creative Commons Attribution 4.0 International
</a>
<br>
<br>
<h3>
<a id="2">
2 SPIDER-T2W-Subset ImageMask Dataset
</a>
</h3>
<h3>2.1 Download SPIDER-T2W-Subset-ImageMask-Dataset</h3>
 If you would like to train this SPIDER-T2W-Subset Segmentation model by yourself,
 please download the dataset from the google drive  
 <a href="https://drive.google.com/file/d/1MabgzvEJy2NBbKzWYFVMfo3t2WsWAewo/view?usp=sharing">
SPIDER-T2W-ImageMask-Subset.zip</a> (
<a href="https://creativecommons.org/licenses/by/4.0/legalcode">
Creative Commons Attribution 4.0 International
</a>)
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<br>
<pre>
./dataset
└─SPIDER-T2W-Subset
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>SPIDER-T2W-Subset Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/SPIDER-T2W-Subset_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not so large to use for the
 training set of our segmentation model.
<br>
<h3>2.2 Derivation of T2W ImageMask-Subset</h3>
The folder structure of the SPIDER is the following. 
<pre>
./SPIDER
 ├─images
 │   ├─1_t1.mha
 │   ├─1_t2.mha
 ...
 │   ├─257_t1.mha
 │   └─257_t2.mha
 └─masks
      ├─1_t1.mha
      ├─1_t2.mha
 ...
      ├─257_t1.mha
      └─257_t2.mha
      
</pre>
We used a simple Python script and the following <b>class-color-mapping-table</b> to generate our 
PNG dataset with colorized masks from the pairs of volumes <b>"images/*t2.mha"</b> and
<b>"masks/*t2.mha"</b> of SPIDER.<br><br>
<a id="color-class-mapping-table"><b>SPIDER class color mapping table</b></a>
<br><br>
<table border=1 style='border-collapse:collapse;' cellpadding='5'>
<tr><th>Indexed Color</th><th>Color</th><th>RGB</th><th>Class</th></tr>
<tr><td>0</td><td with='80' height='auto'><img src='./color_class_mapping/Backgroumd.png' widith='40' height='25'></td><td>(0, 0, 0)</td><td>Backgroumd</td></tr>
<tr><td>1</td><td with='80' height='auto'><img src='./color_class_mapping/Vertebra L5.png' widith='40' height='25'></td><td>(255, 0, 0)</td><td>Vertebra L5</td></tr>
<tr><td>2</td><td with='80' height='auto'><img src='./color_class_mapping/Vertebra L4.png' widith='40' height='25'></td><td>(0, 255, 0)</td><td>Vertebra L4</td></tr>
<tr><td>3</td><td with='80' height='auto'><img src='./color_class_mapping/Vertebra L3.png' widith='40' height='25'></td><td>(0, 0, 255)</td><td>Vertebra L3</td></tr>
<tr><td>4</td><td with='80' height='auto'><img src='./color_class_mapping/Vertebra L2.png' widith='40' height='25'></td><td>(255, 255, 0)</td><td>Vertebra L2</td></tr>
<tr><td>5</td><td with='80' height='auto'><img src='./color_class_mapping/Vertebra L1.png' widith='40' height='25'></td><td>(0, 255, 255)</td><td>Vertebra L1</td></tr>
<tr><td>6</td><td with='80' height='auto'><img src='./color_class_mapping/Vertebra T12.png' widith='40' height='25'></td><td>(255, 0, 255)</td><td>Vertebra T12</td></tr>
<tr><td>7</td><td with='80' height='auto'><img src='./color_class_mapping/Vertebrae (Upper).png' widith='40' height='25'></td><td>(128, 40, 40)</td><td>Vertebrae (Upper)</td></tr>
<tr><td>8</td><td with='80' height='auto'><img src='./color_class_mapping/Spinal Canal.png' widith='40' height='25'></td><td>(40, 128, 40)</td><td>Spinal Canal</td></tr>
<tr><td>9</td><td with='80' height='auto'><img src='./color_class_mapping/Partially visible Vertebrae.png' widith='40' height='25'></td><td>(40, 40, 128)</td><td>Partially visible Vertebrae</td></tr>
<tr><td>10</td><td with='80' height='auto'><img src='./color_class_mapping/Intervertebral Disc L5-S1.png' widith='40' height='25'></td><td>(128, 128, 0)</td><td>Intervertebral Disc L5-S1</td></tr>
<tr><td>11</td><td with='80' height='auto'><img src='./color_class_mapping/Intervertebral Disc L4-L5.png' widith='40' height='25'></td><td>(0, 128, 128)</td><td>Intervertebral Disc L4-L5</td></tr>
<tr><td>12</td><td with='80' height='auto'><img src='./color_class_mapping/Intervertebral Disc L3-L4.png' widith='40' height='25'></td><td>(128, 0, 128)</td><td>Intervertebral Disc L3-L4</td></tr>
<tr><td>13</td><td with='80' height='auto'><img src='./color_class_mapping/Intervertebral Disc L2-L3.png' widith='40' height='25'></td><td>(40, 40, 40)</td><td>Intervertebral Disc L2-L3</td></tr>
<tr><td>14</td><td with='80' height='auto'><img src='./color_class_mapping/Intervertebral Disc L1-L2.png' widith='40' height='25'></td><td>(80, 80, 80)</td><td>Intervertebral Disc L1-L2</td></tr>
<tr><td>15</td><td with='80' height='auto'><img src='./color_class_mapping/Intervertebral Disc T12-L1.png' widith='40' height='25'></td><td>(128, 128, 128)</td><td>Intervertebral Disc T12-L1</td></tr>
<tr><td>16</td><td with='80' height='auto'><img src='./color_class_mapping/Intervertebral Discs (Upper).png' widith='40' height='25'></td><td>(255, 255, 255)</td><td>Intervertebral Discs (Upper)</td></tr>
</table>
<br><br>
For simplicity, we excluded all empty black mask and their corresponding image slices of the MHA volumes,
because they were irrelevant to train our segmentation model.<br>
Furthermore, we removed image and mask slices whose width was greater than their height to generate a small subset 
from the original dataset.
<br><br>
<h3>2.3 Train Sample Images and Nasks</h3>
.
<b>Train sample images</b><br>
<img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train sample masks</b><br>
<img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained SPIDER-T2W-Subset TensorFlowFlexUNet Model by using the 
<a href="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16 </b> and large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large <b>num_layers=8</b> (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = TensorFlowFlexUNet"
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False
num_classes    = 17
base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b>RGB Color map</b><br>
Specifed rgb color map dict for SPIDER-T2W-Subset 1+16 classes.<br>
<pre>
[mask]
mask_datatyoe    = "categorized"
mask_file_format = ".png"
;SPIDER rgb color map dict for 1+16 classes.
rgb_map = {(0,0,0):0, (255,0,0):1, (0,255,0):2, (0,0,255):3, (255,255,0):4, (0,255,255):5, (255,0,255):6, (128,40,40):7,(40, 128,40):8,\
        (40,40,128):9, (128,128,0):10, (0,128,128):11,(128,0,128):12, (40,40,40):13, (80,80, 80):14, (128,128,128):15, (255,255,255):16}
                  
</pre>
<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>
By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> 
<br> 
As shown below, early in the model training, the predicted masks from our UNet segmentation model showed 
discouraging results.
 However, as training progressed through the epochs, the predictions gradually improved. 
 <br> 
<br>
<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 22,23,24)</b><br>
<img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/asset/epoch_change_infer_at_middle.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 45,46,47)</b><br>
<img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was stopped at epoch 47 by an EarlyStoppintCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/asset/train_console_output_at_epoch47.png" width="1024" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for SPIDER-T2W-Subset.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/asset/evaluate_console_output_at_epoch47.png" width="1024" height="auto">
<br><br>Image-Segmentation-SPIDER-T2W-Subset

<a href="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this <b>SPIDER-T2W-Subset/test</b> was low and dice_coef_multiclass high as shown below.
<br>
<pre>
categorical_crossentropy,0.0334
dice_coef_multiclass,0.9842
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowUNet model for SPIDER-T2W-Subset.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of SPIDER-T2W-Subset Images </b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the 
ground truth masks.
<br><br>
<a href="#color-class-mapping-table">Class color mapping table</a>
<br><br>
<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test/images/1007_5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test/masks/1007_5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_output/1007_5.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test/images/1089_4.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test/masks/1089_4.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_output/1089_4.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test/images/1079_4.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test/masks/1079_4.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_output/1079_4.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test/images/1166_12.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test/masks/1166_12.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_output/1166_12.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test/images/1176_5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test/masks/1176_5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_output/1176_5.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test/images/1200_6.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test/masks/1200_6.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_output/1200_6.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<br>
<h3>
6 3D Volume Segmentation
</h3>
Please move <b>./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset</b> folder, and run the following bat file to segment
2D slices of 3D volume MHA files
 by the Trained-TensorFlowFlexUNet model for SPIDER-T2W-Subset.<br>
<pre>
>./5.infer3d.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNet3DInferencer.py ./train_eval_infer.config
</pre>

<b>infer3d section </b> in <a href="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/train_eval_infer.config">
train_eval_infer.config
<a></b>
<pre>
[infer3d] 
;Specify an images_dir which contains NIfTI or MHA files
images_dir    = "./mini_test_3d/images/"
output_dir    = "./mini_test_3d_output/"
slice_shape_order = "hwd"
slice_normalize = True
slice_resize   = None
slice_rotation = None
mask_overlay  = True
</pre>
<hr>
<b>Acutual Image Segmentation for 2D Slices of a SPIDER-T2W-Subset MHA</b><br>
Some Slices, Inferred Masks and Mask overlays for a 3D volume <b>250_t2.mha</b> file.<br>
<br>
<a href="#color-class-mapping-table">Class color mapping table</a>
<br><br>
<table>
<tr>
<th>Input: Slice</th>
<th>Prediction: Inferred mask</th>
<th>Mask Overlay</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_3d_output/250_t2.mha/slices/10002.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_3d_output/250_t2.mha/masks/10002.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_3d_output/250_t2.mha/overlays/10002.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_3d_output/250_t2.mha/slices/10004.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_3d_output/250_t2.mha/masks/10004.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_3d_output/250_t2.mha/overlays/10004.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_3d_output/250_t2.mha/slices/10007.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_3d_output/250_t2.mha/masks/10007.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_3d_output/250_t2.mha/overlays/10007.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_3d_output/250_t2.mha/slices/10009.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_3d_output/250_t2.mha/masks/10009.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_3d_output/250_t2.mha/overlays/10009.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_3d_output/250_t2.mha/slices/10011.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_3d_output/250_t2.mha/masks/10011.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_3d_output/250_t2.mha/overlays/10011.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_3d_output/250_t2.mha/slices/10013.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_3d_output/250_t2.mha/masks/10013.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/mini_test_3d_output/250_t2.mha/overlays/10013.png" width="320" height="auto"></td>

</tr>
</table>
<hr>
<br>
<br>
<h3>
7 MaskOverlay Video of 3D Volume Segmentation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset</b> folder, and run the following bat file 
to generate <b>overlays.mp4</b> or <b>overlay.gif</b> for MaskOverlays of 3D Volume Segmentation. <br>
<pre>
>./6.video3d.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/MaskOverlayVideoGenerator.py ./train_eval_infer.config
</pre>
<br>

<b>infer3d section </b> in <a href="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/train_eval_infer.config">
train_eval_infer.config
<a></b>

<pre>
[infer3d] 
mask_overlay  = True
;Specify ".mp4" or ".gif".
;video_fileformat  = ".mp4"
video_fileformat  = ".gif"
</pre>
<br>
<b>overlays.gif</b><br>
<img src="./projects/TensorFlowFlexUNet/SPIDER-T2W-Subset/video_3d/overlays.gif" width="520" height="520">
<br>
<br>
<h3>
References
</h3>
<b>1. Lumbar spine segmentation in MR images: a dataset and a public benchmark</b><br>
Jasper W. van der Graaf, Miranda L. van Hooff, Constantinus F. M. Buckens, Matthieu Rutten, Job L. C. van Susante, <br>
Robert Jan Kroeze, Marinus de Kleuver, Bram van Ginneken & Nikolas Lessmann<br>
<a href="https://www.nature.com/articles/s41597-024-03090-w">
https://www.nature.com/articles/s41597-024-03090-w</a>
<br>
<br>
<b>2. Lumbar spine segmentation method based on deep learning</b><br>
Hongjiang Lu, Mingying Li, Kun Yu, Yingjiao Zhang, Liang Yu<br>
<a href="https://aapm.onlinelibrary.wiley.com/doi/10.1002/acm2.13996">
https://aapm.onlinelibrary.wiley.com/doi/10.1002/acm2.13996</a>
<br><br>
<b>3. Automatic semantic segmentation of the lumbar spine: Clinical applicability in a multi-parametric and multi-center study on magnetic resonance images</b><br>
Jhon Jairo Sáenz-Gamboa, Julio Domenech, Antonio Alonso-Manjarrés, Jon A. Gómez, Maria de la Iglesia-Vayá <br>
<a href="https://www.sciencedirect.com/science/article/pii/S0933365723000738">
https://www.sciencedirect.com/science/article/pii/S0933365723000738</a>
<br><br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-Lumbar-Spine</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Lumbar-Spine">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Lumbar-Spine</a>
<br><br>
<b>5. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model</a>
<br><br>

