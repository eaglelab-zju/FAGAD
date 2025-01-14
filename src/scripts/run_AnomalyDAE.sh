# nohup bash scripts/run_AnomalyDAE.sh 4 >logs/AnomalyDAE_fb.log &
gpu=$1


# for d in "Facebook" "YelpChi"; do
for d in "YelpChi"; do
    for embed_dim in 64; do
    # 256 128 OOM
        for lr in 0.01 0.001 0.0001; do
            for dropout in 0.2 0.5 0.8; do
                for alpha in 0.1 0.5 0.9; do
                    python -u -m dgld.train.main --dataset $d --device $gpu --runs 3 AnomalyDAE --embed_dim $embed_dim --out_dim $(($embed_dim/2)) --lr $lr --dropout $dropout --alpha $alpha --weight_decay 0.00001 --num_epoch 100
done
    done
        done
            done
                done
