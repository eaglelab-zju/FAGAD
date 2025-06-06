# FAGAD

An implementation of FAGAD: [Frequency Self-Adaptation Graph Neural Network for Unsupervised Graph Anomaly Detection](https://www.sciencedirect.com/science/article/pii/S0893608025004927).

## Installation

See [requirements-dev.txt](./requirements-dev.txt) and [requirements.txt](./requirements.txt) for pip requirements.

Installation scripts:
```bash
bash .ci/install-dev.sh
bash .ci/install.sh
```

## Run Model

See [sota.sh](./src/scripts/sota.sh) for running scripts and [baselines](./src/scripts) for baseline tuning scripts.

```bash
source .env/bin/activate
cd src

# eta is the beta in paper

python -u -m dgld.models.FAGAD.models --dataset YelpChi --gpu 7 --model_init zero --lr 0.001 --hid_feats 64 --mlp_hidden_dic 64 --projection_dic 64 --struct_dec_act relu --k_dic 2 --alpha 0.2 --eta 0.5 --num_epoch 30 --dropout_dic 0.1 --runs 3 --weight_decay 0

python -u -m dgld.models.FAGAD.models --dataset Facebook --gpu 7 --model_init zero --lr 0.001 --hid_feats 2048  --mlp_hidden_dic 2048 --projection_dic 2048 --struct_dec_act relu --k_dic 8 --alpha 0.1 --eta 0.9 --num_epoch 100 --dropout_dic 0 --runs 3 --weight_decay 0.00001

python -u -m dgld.models.FAGAD.models --dataset Amazon --gpu 7 --model_init zero --lr 0.001 --hid_feats 512 --mlp_hidden_dic 512  --projection_dic 512 --struct_dec_act relu --k_dic 1 --alpha 0.2 --eta 0.9 --num_epoch 20 --dropout_dic 0 --runs 3 --weight_decay 0.00001

python -u -m dgld.models.FAGAD.models --dataset reddit --gpu 7 --model_init zero --lr 0.0001 --hid_feats 128 --mlp_hidden_dic 1024 --projection_dic 128 --struct_dec_act sigmoid --k_dic 2 --alpha 0.99 --eta 0.5 --num_epoch 20 --dropout_dic 0 --runs 3 --weight_decay 0.00001

python -u -m dgld.models.FAGAD.models --dataset wiki --gpu 7 --model_init zero --lr 0.00001 --hid_feats 1024 --mlp_hidden_dic 2048 --projection_dic 1024 --struct_dec_act relu --k_dic 5 --alpha 0.001 --eta 0.9 --num_epoch 25 --dropout_dic 0 --runs 3 --weight_decay 0.00001

python -u -m dgld.models.FAGAD.models --dataset Enron --gpu 5 --model_init identity --lr 0.001 --hid_feats 2048 --mlp_hidden_dic 256 --projection_dic 2048 --struct_dec_act sigmoid --k_dic 2 --alpha 0.999 --eta 0.5 --num_epoch 5   --dropout_dic 0 --runs 3 --weight_decay 0

python -u -m dgld.models.FAGAD.models --dataset BlogCatalog --gpu 5 --model_init zero --lr 0.001 --hid_feats 512 --mlp_hidden_dic 1024  --projection_dic 1024 --struct_dec_act sigmoid --k_dic 5 --alpha 0.99 --eta 0.1 --num_epoch 25 --weight_decay 0 --dropout_dic 0.1 --runs 3

python -u -m dgld.models.FAGAD.models --dataset Flickr --gpu 4 --model_init zero --lr 0.001 --hid_feats 512  --mlp_hidden_dic 768  --projection_dic 512  --struct_dec_act relu --k_dic 2 --alpha 0.99 --eta 0.5 --num_epoch 15 --dropout_dic 0.2 --weight_decay 0 --runs 3
```

## Citation
```bibtex
@article{gu2025frequency,
  title={Frequency Self-Adaptation Graph Neural Network for Unsupervised Graph Anomaly Detection},
  author={Gu, Ming and Yang, Gaoming and Zheng, Zhuonan and Liu, Meihan and Wang, Haishuai and Chen, Jiawei and Zhou, Sheng and Bu, Jiajun},
  journal={Neural Networks},
  pages={107612},
  year={2025},
  publisher={Elsevier}
}
```
