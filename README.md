# Drainage

## Requirements

```console
python >= 3.9, torch >= 1.12.1, torchvision >= 0.13.1, numpy >= 1.23.1
```

## How to use

### Configs

- Check '*.json' file in the config folder for each exeriment.
- update DATA_DIR in main.pyto point to cifar-10, cifar-100
- update CIFAR10_HUMAN_NOISE_PATH and CIFAR100_HUMAN_NOISE_PATH in dataset.py to point to human-annotated noise files.
- update webvision configs: train_data_path and val_data_path to point to webvisoin train and validation set folders.
   - The val_data_path could also point to ILSVRC2012 validation set folder for evaluation.
-update all clothing1m configs: data_path to point to clothing1m dataset folder.

### Arguments

* gpu: GPU id
* seed: random seed
* config: config name
* noise_type: 'asym' if use asymmetric noise, 'instance' if use instance-dependent noise, 'human' if use human-annotated noise. if human, noise_rate is ignored.
* noise_rate: noise rate; between 0 and 1
* eval_freq: frequency of evaluation, default is 1
* tuning: use the tuning settings (80% of the original training set as training set and 20% as validation set)

### Example

Training ANL-CE on CIFAR-10 with 0.8 symmetric noise:
```bash
python main.py \
--gpu 0 \
--seed 1 \
--config cifar10_alpha_dl \ 
--noise_type asym \
--noise_rate 0.45 \
--eval_freq 10
```

## Thanks

Moreover, we thank the codes implemented by [Ye et al.](https://github.com/Virusdoll/Active-Negative-Loss), [Ma et al.](https://github.com/HanxunH/Active-Passive-Losses) and [Zhou et al.](https://github.com/hitcszx/ALFs).