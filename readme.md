# [NeurIPS 2026] Interval-based Task Activation Consolidation

<div align="center">
  <div>
    Patryk Krukowski<sup>1,2,*</sup>&emsp;
    Jan Miksa<sup>1,3</sup>&emsp;
    Piotr Helm<sup>1</sup>&emsp;
    Jacek Tabor<sup>4,5</sup>&emsp;
    Paweł Wawrzyński<sup>3</sup>&emsp;
    Przemysław Spurek<sup>3,4</sup>
  </div>
  <br>
  <div>
    <sup>1</sup>Doctoral School of Exact and Natural Sciences, Jagiellonian University, Kraków, Poland
    <br>
    <sup>2</sup>Akces NCBR
    <br>
    <sup>3</sup>IDEAS Research Institute
    <br>
    <sup>4</sup>Jagiellonian University, Kraków, Poland
    <br>
    <sup>5</sup>Centre for Credible AI
    <br><br>
    <sup>*</sup>Corresponding author: patryk.krukowski@doctoral.uj.edu.pl
  </div>
</div>

</div>
</div>
<p align="center">
  <a href='https://arxiv.org/abs/2505.04575'><img src='https://img.shields.io/badge/Arxiv-2505.04575-A42C25.svg?logo=arXiv'></a>
  <a href='[https://arxiv.org/abs/2505.04575](https://github.com/zhoujiahuan1991/ICML2025-KA-Prompt)'><img src="https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fzhoujiahuan1991%2FICML2025-KA-Prompt&label=KA-Prompt&icon=github&color=%233d8bfd"></a>
</p>

The *official* repository for  [Componential Prompt-Knowledge Alignment for Domain Incremental Learning](https://openreview.net/pdf?id=QWCdBzLOsk).

![Framework](figs/framework.png)


## Installation
```shell
conda create -n intact_with_ka_prompt python=3.9
conda activate intact_with_ka_prompt
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```


### Data and Model Preparation

This work primarily utilizes DomainNet, ImageNet-R, ImageNet-C, and ImageNet-Mix. Among them, DomainNet, ImageNet-R, and ImageNet-C are existing datasets. ImageNet-Mix is a dataset we constructed based on ImageNet-R and ImageNet-C. And you can download DomainNet at [here](https://ai.bu.edu/M3SDA/), ImageNet-R at [here](https://github.com/hendrycks/imagenet-r?tab=readme-ov-file) and ImageNet-C at [here](https://github.com/hendrycks/robustness?tab=readme-ov-file). To facilitate your use of our code, we have included the training-testing sample split files for each domain of each dataset in ./dataset/image_list/.




Then your data directory should be organized in the following format:

- **you_data_path**
  - *DomainNet*
  - *ImageNet-R*
  - *ImageNet-C*

The pre-trained model weights can be downloaded at [here](https://huggingface.co/google/vit-base-patch16-224-in21k).


### KA-Prompt
You can directly run the pre-written shell script:
```
sh scripts/DomainNet.sh
sh scripts/ImageNet-R.sh
sh scripts/ImageNet-C.sh
sh scripts/ImageNet-Mix.sh
```



## Results
![Results](figs/results.png)

## Citation
### If you find this code useful for your research, please cite our paper.

@article{xu2025comp,
  title={Componential Prompt-Knowledge Alignment for Domain Incremental Learning},
  author={Xu, Kunlun and Zou, Xu and Hua, Gang and Zhou, Jiahuan},
  booktitle={ICML}, 
  year={2025}
}

## Acknowledgement
Our code is based on the PyTorch implementation of [C-Prompt](https://github.com/zhoujiahuan1991/IJCV2024-C-Prompt). 

## Contact

For any questions, feel free to contact us (xkl@stu.pku.edu.cn).

Welcome to our Laboratory Homepage ([OV<sup>3</sup> Lab](https://zhoujiahuan1991.github.io/)) for more information about our papers, source codes, and datasets.
