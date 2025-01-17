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
--latent-size-scale 4 \ # the latent embedding size (c) of the read-write attention QK-matrices model. This reduces the embedding dimension of the QK matrices to c = d/r, where r is the scale set by this flag.
--fusion-type "residual" 
--process-embedded-type patch \
--drop-path-rate 0.0 \
--checkpoints-path "<PATH-TO-SAVE-CHECKPOINTS>" \
--use-mixup \
--mixup-alpha 0.5 \
--image-size 224 \
--compile \ # torch.compile enable flag.
# --wandb \ # weights and biases logging.
```


## ImageNet-1K Fine-tuning
We finetune on ImageNet-1K. Please refer to [training configs](#configs) for training configurations.

```sh
python train_lt.py \
--devices 2 \ # number of gpus
--num-nodes 1 \ # number of nodes 
--dataset "imagenet" \
--data-path "<PATH-TO-IMAGENET-1k>" \
--num-workers 32 \
--pin-memory \
--num-classes 1000 \
--epochs 300 \
--warmup-steps 3000 \
--batch-size 1024 \
--test-batch-size 64 \
--accumulations 1 \
--scheduler cosine \
--learning-rate 0.25e-4 \
--warmup-lr 1.0e-6 \
--min-lr 1.0e-7 \
--weight-decay 0.1 \
--grad-clip 1.00 \
--model "vittm_base" \
--rand-aug \ # used to enable rand-aug
--num-ops 3 \ # number of augmentations 
--magnitude 20 \ # magnitude of augmentions
--memory-ps 28 \ # memory patch size
--process-ps 28 \ # process patch size
--rw-head-type "lin" \ # rw-head that is used
--latent-size-scale 4 \ # the latent embedding size (c) of the read-write attention QK-matrices model. This reduces the embedding dimension of the QK matrices to c = d/r, where r is the scale set by this flag.
--fusion-type "residual" \
--process-embedded-type "patch" \
--drop-path-rate 0.20 \ # stochastic path drop.
--gradnorm \
--checkpoint-profile vittm-base-21k-300-im1k-300-lin-ls2 \
--use-cutmix \ # enable cut-mix
--use-mixup \ # enable mixup
--mixup-alpha 0.8 \ # mixup and cutmix 
--random-erasing \ # random erasing
--bce \ # binary cross entropy
--compile \
# --checkpoints-path "<PATH-TO-SAVE-CHECKPOINTS>" \
# --wandb \
```

## Configs

### Enchancements for Improving Performance
Although the original model is trained using the AdamW optimizer, swapping to the Shampoo optimizer is much better and can result in stronger models. 
Feel free to try it out. 
Please look at [train_lt.py](train_lt.py#L106) --- uncomment [L34-35](train_lt.py#L34)

### Pretraining Configurations

|                       | ViTTM-S      | ViTTM-B        |
| --------------------- | ------------ | -------------- |
| dataset               | imagenet-21k | "imagenet-21k" |
| num-classes           | 10450        | 10450          |
| epochs                | 300          | 300            |
| warmup-steps          | 10000        | 10000          |
| effective-batch-size  | 4096         | 4096           |
| scheduler             | cosine       | cosine         |
| learning-rate         | 1.50e-4      | 1.50e-4        |
| warmup-lr             | 1.0e-6       | 1.0e-6         |
| min-lr                | 1e-7         | 1e-7           |
| weight-decay          | 3.0e-2       | 3.0e-2         |
| grad-clip             | 1.00         | 1.00           |
| rand-aug              | ✓            | ✓              |
| num-ops               | 2            | 2              |
| magnitude             | 15           | 15             |
| memory-ps             | 28           | 28             |
| process-ps            | 28           | 28             |
| rw-head-type          | lin          | lin            |
| latent-size-scale     | 4            | 4              |
| fusion-type           | residual     | residual       |
| process-embedded-type | patch        | patch          |
| drop-path-rate        | 0.0          | 0.0            |
| mixup                 | ✓            | ✓              |
| mixup-alpha           | 0.5          | 0.5            |
| image-size            | 224          | 224            |



### Fine-tuning Configurations
| Flag                  | ViTTM-S  | ViTTM-B  |
| --------------------- | -------- | -------- |
| dataset               | imagenet | imagenet |
| num-classes           | 1000     | 1000     |
| epochs                | 300      | 300      |
| warmup-steps          | 3000     | 3000     |
| effective-batch-size  | 4096     | 2048     |
| scheduler             | cosine   | cosine   |
| learning-rate         | 1.75e-4  | 0.25e-4  |
| warmup-lr             | 1.0e-6   | 1.0e-6   |
| min-lr                | 1.0e-6   | 1.0e-7   |
| weight-decay          | 3.0e-2   | 0.1      |
| grad-clip             | 1.00     | 1.00     |
| rand-aug              | ✓        | ✓        |
| num-ops               | 2        | 3        |
| magnitude             | 15       | 20       |
| memory-ps             | 28       | 28       |
| process-ps            | 28       | 28       |
| rw-head-type          | lin      | lin      |
| latent-size-scale     | 4        | 4        |
| fusion-type           | residual | residual |
| process-embedded-type | patch    | patch    |
| drop-path-rate        | 0.1      | 0.20     |
| cutmix                | ✓        | ✓        |
| mixup                 | ✓        | ✓        |
| mixup-alpha           | 0.5      | 0.8      |
| random-erasing        | ✓        | ✓        |
| bce                   | ✓        | ✓        |
| image-size            | 224      | 224      |

### Description of Hyperparameters
| Flag                  | Description                               |
| --------------------- | ----------------------------------------- |
| dataset               | Dataset name                              |
| num-classes           | Number of classes in ImageNet-1k          |
| epochs                | Total number of training epochs           |
| warmup-steps          | Number of steps for learning rate warm-up |
| effective-batch-size  | Effective batch size after accumulation   |
| scheduler             | Learning rate scheduler type              |
| learning-rate         | Initial learning rate                     |
| warmup-lr             | Learning rate during warm-up              |
| min-lr                | Minimum learning rate                     |
| weight-decay          | Weight decay                              |
| grad-clip             | Gradient clipping threshold               |
| rand-aug              | Enable RandAugment                        |
| num-ops               | Number of augmentation operations         |
| magnitude             | Magnitude of augmentations                |
| memory-ps             | Memory patch size                         |
| process-ps            | Process patch size                        |
| rw-head-type          | Type of read-write head                   |
| latent-size-scale     | Scale for latent embedding size           |
| fusion-type           | Type of fusion for model layers           |
| process-embedded-type | Type of processing for embedded data      |
| drop-path-rate        | Stochastic depth rate                     |
| cutmix                | Enable CutMix augmentation                |
| mixup                 | Enable Mixup augmentation                 |
| mixup-alpha           | Alpha parameter for Mixup                 |
| random-erasing        | Enable Random Erasing augmentation        |
| bce                   | Use Binary Cross Entropy loss             |
| image-size            | Size of input images                      |

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