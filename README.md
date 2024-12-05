<div align="center">

# WACV 2025: Token Turing Machines are Efficient Vision Models

[![arXiV](https://img.shields.io/badge/arXiV-611111?logo=arxiv)](https://arxiv.org/abs/2409.07613v1)

</div>

<div align="center">

This repo contains software artifacts for the WACV 2025 Paper *Token Turing Machines are Efficient Vision Models.*

Feel free to submit pull requests!

</div>

# Environment Setup 

```sh
conda create -n vttm python=3.11
pip install -r requirements.txt
```

# Training
## ImageNet-21k-P Pre-Training
We pretrain using the Winter 21 version of the [ImageNet-21K-P dataset](https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/dataset_preprocessing/processing_instructions.md). Please refer to [training configs](#configs) for training configurations.

```sh
python train_lt.py \
--devices 2 \ # number of gpus
--num-nodes 2 \ # number of nodes 
--dataset "imagenet-21k" \
--data-path "<PATH-TO-IMAGENET-21k>" \
--num-workers 32 \
--pin-memory \
--num-classes 10450 \ # 21k-P consists of 10450 classes
--epochs 300 \ 
--warmup-steps 10000 \
--batch-size 1024 \
--test-batch-size 256 \
--accumulations 1 \
--scheduler cosine \
--learning-rate 1.50e-4 \
--warmup-lr 1.0e-6 \
--min-lr 1e-7 \
--weight-decay 3.0e-2 \
--grad-clip 1.00 \
--model "vittm_base" \
--rand-aug \ # used to enable rand-aug
--num-ops 2 \ # number of augmentations 
--magnitude 15 \ # magnitude of augmentions
--memory-ps 28 \ # memory patch size
--process-ps 28 \ # process patch size
--rw-head-type "lin" \ # rw-head that is used
--latent-size-scale 4 \
--fusion-type "residual" 
--process-embedded-type patch \
--drop-path-rate 0.0 \
--checkpoints-path "<PATH-TO-SAVE-CHECKPOINTS>" \
--use-mixup \
--mixup-alpha 0.5 \
--image-size 224 \
--compile \ # torch.compile setting.
# --wandb \ # weights and biases logging.
```


## ImageNet-1K Fine-tuning

## Configs

### Pretraining Configurations

### Fine-tuning Configurations

<!-- Citation -->
# Citation 
## BibTeX
```bib
@misc{jajal2024tokenturingmachinesefficient,
      title={Token Turing Machines are Efficient Vision Models}, 
      author={Purvish Jajal and Nick John Eliopoulos and Benjamin Shiue-Hal Chou and George K. Thiravathukal and James C. Davis and Yung-Hsiang Lu},
      year={2024},
      eprint={2409.07613},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.07613}, 
}
```