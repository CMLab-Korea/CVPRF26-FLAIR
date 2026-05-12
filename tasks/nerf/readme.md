# INR_NeRF

A Neural Radiance Field (NeRF) implementation enhanced with custom activation functions such as **RC-Gauss**, and supporting CUDA acceleration for high-performance rendering.

---

## 🚀 Version Info

- **PyTorch**: 1.10.0  
- **CUDA**: 11.3 (cu113)  
- **Python**: 3.9 (recommended)  

---

## 📦 Installation

Install the required dependencies via:

```bash
pip install -r requirements.txt
```


## 📂 Dataset

This project uses the **NeRF-synthetic** dataset for training and evaluation.

You can download the dataset from one of the following sources:

- 🔗 [Kaggle: nerf-synthetic-dataset](https://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset)
- 📁 Shared Drive: `IntelxCMLab > [INR] > Dataset > NeRF > drums`

Once downloaded, make sure to specify the correct path using the `--path` argument when running the training command:

```bash
--path /your/path/to/nerf_synthetic/drums
```


## 💻 Example Training Command

To train on the **NeRF-synthetic** `drums` scene using the `bla` network, run the following command:

```bash
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_nerf.py --path /home/torch-ngp/data/nerf_synthetic/drums --nn bla --lr 2e-4 --iter 50000 --downscale 4 --trainskip 4 --num_layers 3 --hidden_dim 64 --geo_feat_dim 64 --num_layers_color 3 --hidden_dim_color 64 --init_T 1.0 --init_beta 0.05 --init_zeta 0.1 --sigma 30.0 --trainable True --cuda_ray --workspace logs/drums_bla_small --bound 1 --scale 0.8 --dt_gamma 0
```
