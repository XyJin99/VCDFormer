# VCDFormer (JAG 2025)
### 📖[**Paper**](https://www.sciencedirect.com/science/article/pii/S1569843225001128) | 🖼️[**PDF**](/figures/VCDFormer.pdf)

PyTorch codes for "[VCDFormer: Investigating Cloud Detection Approaches in Sub-Second-Level Satellite Videos](https://www.sciencedirect.com/science/article/pii/S1569843225001128)", **International Journal of Applied Earth Observation and Geoinformation (JAG)**, 2025.

Authors: Xianyu Jin, [Jiang He*](https://jianghe96.github.io/), [Yi Xiao](https://xy-boy.github.io/), Ziyang Lihe, Jie Li, and [Qiangqiang Yuan*](https://www.sgg.whu.edu.cn/info/1425/2104.htm) <br>
Wuhan University and Technical University of Munich

### Abstract
>Satellite video, as an emerging data source for Earth observation, enables dynamic monitoring and has wide-ranging applications in diverse fields. Nevertheless, cloud occlusion hinders the ability of satellite video to provide uninterrupted monitoring of the Earth’s surface. To mitigate the interference of clouds, cloud-free areas need to be selected before application, or an optimized solution like a cloud removal algorithm can be utilized to recover the occluded regions, both of which inherently demand the precise detection of clouds. However, no existing methods are capable of robust cloud detection in satellite videos. We propose the first sub-second-level satellite video cloud detection model VCDFormer to handle this problem. In VCDFormer, a spatial-temporal-enhanced transformer consisting of a local spatial-temporal reconfiguration block and a spatial-enhanced block is introduced to explore global spatial-temporal correspondence efficiently. Additionally, we construct WHU-VCD, the first sub-second-level synthetic dataset specifically designed to capture the more realistic motion characteristics of both thick and thin clouds in satellite videos. Compared to the state-of-the-art cloud detection methods, VCDFormer achieves an approximate 10–15% improvement in the IoU metric and a 5–8% increase in the F1-Score on the simulated test set. Experimental evaluations on Jilin-1 satellite videos, involving both synthetic and real-world scenarios, demonstrate that our proposed VCDFormer achieves superior performance in satellite video cloud detection tasks. The source code is available at https://github.com/XyJin99/VCDFormer.

### Overall
 ![image](/figures/network.png)

## Dataset Preparation
We created a synthetic dataset named WHU-VCD based on the atmospheric scattering model with authentic cirrus cloud bands and satellite video scenes to simulate the motion characteristics of both thick and thin clouds in sub-second-level satellite videos.
Please download our dataset in 
 * Baidu Netdisk [WHU-VCD](https://pan.baidu.com/s/1sCXvKb_3HKq0xtvYx8y5Zg) Code: 5nhm
 * Zenodo: TODO

You can also train your dataset following the directory structure below!
 
### Data directory structure
train--  
&emsp;|&ensp;cloud--  
&emsp;&emsp;|&ensp;SCENE 000---  
&emsp;&emsp;&emsp;| 00000000.png  
&emsp;&emsp;&emsp;| ···.png  
&emsp;&emsp;&emsp;| 00000099.png     
&emsp;&emsp;|&ensp;SCENE N---  
&emsp;|&ensp;mask--  

eval--  
&emsp;|&ensp;cloud--  
&emsp;&emsp;|&ensp;SCENE 000---  
&emsp;&emsp;&emsp;| 00000000.png  
&emsp;&emsp;&emsp;| ···.png  
&emsp;&emsp;&emsp;| 00000099.png    
&emsp;&emsp;|&ensp;SCENE N---  
&emsp;|&ensp;mask--  

realtest--  
&emsp;|&ensp;cloud--  

## Training
```
python train.py --config=./config/train/train.json
```

## Evaluation
```
python eval.py --config=./config/eval/eval.json
```

## Test
```
python test.py --config=./config/test/test.json
```

### Quantitative results
 ![image](/figures/quantitative.png)
### Qualitative results
 ![image](/figures/qualitative.png)
#### More details can be found in our paper!


## Contact
If you have any questions or suggestions, feel free to contact me. 😊  
Email: jin_xy@whu.edu.cn

## Citation
If you find our work useful in your research, we would appreciate your citation. 😊😊
```
@article{jin2025vcdformer,
  title={VCDFormer: Investigating cloud detection approaches in sub-second-level satellite videos},
  author={Jin, Xianyu and He, Jiang and Xiao, Yi and Lihe, Ziyang and Li, Jie and Yuan, Qiangqiang},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={138},
  pages={104465},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgement
Our work is built upon [SegFormer](https://github.com/NVlabs/SegFormer) and [Swin-Transformer](https://github.com/microsoft/Swin-Transformer).
Thanks to the author for the source code !
