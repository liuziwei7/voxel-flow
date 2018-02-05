# Video Frame Synthesis using Deep Voxel Flow
We address the problem of synthesizing new video frames in an existing video, either in-between existing frames (interpolation), or subsequent to them (extrapolation). This problem is challenging because video appearance and motion can be highly complex. Traditional optical-flow-based solutions often fail where flow estimation is challenging, while newer neural-network-based methods that hallucinate pixel values directly often produce blurry results. We combine the advantages of these two methods by training a deep network that learns to synthesize video frames by flowing pixel values from existing ones, which we call deep voxel flow. Our method requires no human supervision, and any video can be used as training data by dropping, and then learning to predict, existing frames. The technique is efficient, and can be applied at any video resolution. We demonstrate that our method produces results that both quantitatively and qualitatively improve upon the state-of-the-art.

[[Project]](https://liuziwei7.github.io/projects/VoxelFlow) [[Paper]](https://arxiv.org/abs/1702.02463) [[Demo]](https://liuziwei7.github.io/projects/voxelflow/demo.html)      

<img src='./misc/demo.gif' width=810>

## Overview
`Deep Voxel Flow (DVF)` is the author's re-implementation of the video frame synthesizer described in:  
"Video Frame Synthesis using Deep Voxel Flow"   
[Ziwei Liu](https://liuziwei7.github.io/), [Raymond A. Yeh](http://www.isle.illinois.edu/~yeh17/), [Xiaoou Tang](http://www.ie.cuhk.edu.hk/people/xotang.shtml), [Yiming Liu](http://bitstream9.me/), [Aseem Agarwala](http://www.agarwala.org/) (CUHK & UIUC & Google Research)
in International Conference on Computer Vision (ICCV) 2017, Oral Presentation

<img src='./misc/demo_teaser.jpg' width=800>

Further information please contact [Ziwei Liu](https://liuziwei7.github.io/).

## Requirements
* [TensorFlow](https://www.tensorflow.org/)

## Getting started
* Run the training script:
``` bash
python voxel_flow_train.py --subset=train
```
* Run the testing script:
``` bash
python voxel_flow_train.py --subset=test
```
* Run the evaluation script:
``` bash
matlab eval_voxelflow.m
```

## License and Citation
The use of this software is RESTRICTED to **non-commercial research and educational purposes**.

```
@inproceedings{liu2017voxelflow,
 author = {Ziwei Liu, Raymond Yeh, Xiaoou Tang, Yiming Liu, and Aseem Agarwala},
 title = {Video Frame Synthesis using Deep Voxel Flow},
 booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
 month = {October},
 year = {2017} 
}
```

## Disclaimer
This is not an official Google product.
