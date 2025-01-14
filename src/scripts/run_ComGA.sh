# nohup bash scripts/run_ComGA.sh 0 >logs/ComGA_fb.log &
gpu=$1


# for d in "Facebook" "YelpChi"; do
for d in "YelpChi"; do
    for embed_dim in 256 128 64; do
        for lr in 0.01 0.001 0.0001; do
            for dropout in 0.2 0.5 0.8; do
                for alpha in 0.1 0.5 0.9; do
                    python -u -m dgld.train.main --dataset $d --device $gpu --runs 3 ComGA --n_enc_1 $embed_dim --n_enc_2 $(($embed_dim/2)) --n_enc_3 $(($embed_dim/2/2)) --lr $lr --dropout $dropout --alpha $alpha --weight_decay 0.00001 --num_epoch 100
done
    done
        done
            done
                done
