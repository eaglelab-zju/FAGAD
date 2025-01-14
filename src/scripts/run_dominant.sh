# nohup bash scripts/run_dominant.sh 3 >logs/dominant_fb.log &
gpu=$1


# for d in "Facebook" "YelpChi"; do
for d in "YelpChi"; do
    for hidden_dim in 256 128 64; do
        for lr in 0.01 0.001 0.0001; do
            for dropout in 0.2 0.5 0.8; do
                for alpha in 0.1 0.5 0.9; do
                    python -u -m dgld.train.main --dataset $d --device $gpu --runs 3 DOMINANT --hidden_dim $hidden_dim --lr $lr --dropout $dropout --alpha $alpha
done
    done
        done
            done
                done
