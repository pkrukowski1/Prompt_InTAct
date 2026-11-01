# [NeurIPS 2026] InTAct: Interval-based Task Activation Consolidation

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
    <sup>5</sup>Centre for Credible AI, Warsaw University of Technology
    <br><br>
    <sup>*</sup>Corresponding author: patryk.krukowski@doctoral.uj.edu.pl
  </div>
</div>

<p align="center">
  <a href='https://arxiv.org/abs/2511.17439'><img src='https://img.shields.io/badge/Arxiv-2511.17439-A42C25.svg?logo=arXiv'></a>
  <a href='https://github.com/pkrukowski1/Prompt_InTAct'><img src='https://img.shields.io/badge/GitHub-Repo-181717.svg?logo=github'></a>
</p>

The *official* repository for InTAct.

![Framework](figs/framework.png)

## Installation

```shell
conda create -n intact_with_ka_prompt python=3.9
conda activate intact_with_ka_prompt
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install -r requirements.txt
```

### Data and Model Preparation

This work primarily utilizes DomainNet, ImageNet-R, ImageNet-C, and ImageNet-Mix. Among them, DomainNet, ImageNet-R, and ImageNet-C are existing datasets. ImageNet-Mix is a dataset we constructed based on ImageNet-R and ImageNet-C. And you can download DomainNet at [here](https://ai.bu.edu/M3SDA/), ImageNet-R at [here](https://github.com/hendrycks/imagenet-r?tab=readme-ov-file) and ImageNet-C at [here](https://github.com/hendrycks/robustness?tab=readme-ov-file). To facilitate your use of our code, we have included the training-testing sample split files for each domain of each dataset in ./dataset/image_list/.

Then your data directory should be organized in the following format:

- **your_data_path/**
  - *DomainNet/*
  - *ImageNet-R/*
  - *ImageNet-C/*

The pre-trained model weights can be downloaded at [here](https://huggingface.co/google/vit-base-patch16-224-in21k).

### InTAct
You can directly run the pre-written shell script:

```shell
sh scripts/run_imagenet_r.sh
sh scripts/run_imagenet_cr.sh
sh scripts/run_imagenet_c.sh
sh scripts/run_domainnet.sh
```

## Results
![Results](figs/results.png)

## Citation
### If you find this code useful for your research, please cite our paper.

```bibtex
@misc{krukowski2026intactintervalbasedtaskactivation,
      title={InTAct: Interval-based Task Activation Consolidation for Continual Learning}, 
      author={Patryk Krukowski and Jan Miksa and Piotr Helm and Jacek Tabor and Paweł Wawrzyński and Przemysław Spurek},
      year={2026},
      eprint={2511.17439},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={[https://arxiv.org/abs/2511.17439](https://arxiv.org/abs/2511.17439)}
}
```

## Acknowledgement
Our code is based on the PyTorch implementation of [KA-Prompt](https://github.com/zhoujiahuan1991/icml2025-ka-prompt). 

## Contact

For any questions, feel free to contact us (patryk.krukowski@doctoral.uj.edu.pl).