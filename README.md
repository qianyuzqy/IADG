# Instance-Aware Domain Generalization for Face Anti-Spoofing

This is the PyTorch implementation of our paper:

[Paper] Instance-Aware Domain Generalization for Face Anti-Spoofing

Qianyu Zhou, Ke-Yue Zhang, Taiping Yao, Xuequan Lu, Ran Yi, Shouhong Ding, Lizhuang Ma.

The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR), 2023

**[[Arxiv]](https://arxiv.org/pdf/2304.05640.pdf)**
**[[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Instance-Aware_Domain_Generalization_for_Face_Anti-Spoofing_CVPR_2023_paper.pdf)**

## Updates
- (October 2023) All checkpoints of pretrained models are released. 
- (October 2023) All code of IADG are released. 


## Installation

### Requirements

* Linux, CUDA>=11.7, GCC>=5.4
  
* Python>=3.10

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n IADG python=3.10 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate IADG
    ```
  
* PyTorch>=1.13.0, torchvision>=0.14.0 (following instructions [here](https://pytorch.org/)

    For example, if your CUDA version is 11.7, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```


## Usage

### Checkpoints
Below, we provide all checkpoints, all training logs and inference logs of IADG for different datasets.

[DownLoad Link of Google Drive](https://drive.google.com/drive/folders/15QjIXXbatQmXzwtR7pydsB4Jqscm7Vb6?usp=sharing)

[DownLoad Link of Baidu Netdisk](https://pan.baidu.com/s/1a3snwN6O1IUOtxt6VU-t8Q) (password:26xc)


### Training

#### Training on single node
You can use the following training command.  
   
```bash
CUDA_VISIBLE_DEVICES=0 python3 -u -m torch.distributed.launch --nproc_per_node=1 --master_port 17850 ./train.py -c ./configs/ICM2O.yaml
```  

### Evaluation
And then run following command to evaluate it on the testing set:
```bash
CUDA_VISIBLE_DEVICES=0 python3 -u  ./test.py -c ./configs/ICM2O_test.yaml --ckpt checkpoint_file
```

## Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [TFace](https://github.com/Tencent/TFace/tree/master)


## Citing IADG
If you find IADG useful in your research, please consider citing:
```bibtex
@inproceedings{zhou2023instance,
  title={Instance-Aware Domain Generalization for Face Anti-Spoofing},
  author={Zhou, Qianyu and Zhang, Ke-Yue and Yao, Taiping and Lu, Xuequan and Yi, Ran and Ding, Shouhong and Ma, Lizhuang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={20453--20463},
  year={2023}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE), while some 
specific features in this repository are with other licenses. Please refer to 
[LICENSES.md](LICENSES.md) for the careful check, if you are using our code for 
commercial matters.

