#!/usr/bin/env bash
# SDF fitting with FLAIR's BLA on 4 meshes in parallel.
# Run:  cd tasks/sdf && bash train_sdf.sh

set -e
cd "$(dirname "$0")"
PY=/home/cau_jihy/miniconda3/envs/flair/bin/python

for i in 0 1 2 3; do
    case $i in
        0) name=armadillo ;;
        1) name=dragon ;;
        2) name=lucy ;;
        3) name=thai ;;
    esac
    $PY train_sdf.py --model_type bla \
        --config ./configs/bla_${name}.ini \
        --experiment_name ${name}_bla_3x256 \
        --hidden_layers 3 --hidden_size 256 \
        --lr 0.0005 \
        --init_t 1.0 --init_beta 0.05 --init_zeta 1.0 --sigma 30.0 --trainable True \
        --gpu $i &
done
wait
echo "[$(date +%T)] SDF BLA — all 4 meshes done"
