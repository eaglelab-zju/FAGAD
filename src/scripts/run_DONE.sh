# nohup bash scripts/run_DONE.sh 6 >logs/DONE_fb.log &
gpu=$1


# for d in "Facebook" "YelpChi"; do
for d in "YelpChi"; do
    for embedding_dim in 256 128 64; do
        for lr in 0.1 0.01 0.001; do
            for dropout in 0.2 0.5 0.8; do
                for restart in 0 0.5 0.8; do
                    for max_len in 0 3; do
                        python -u -m dgld.train.main --dataset $d --device $gpu --runs 3 DONE --embedding_dim $embedding_dim --lr $lr --dropout $dropout --weight_decay 0.00001 --num_epoch 100 --restart $restart --batch_size 1024 --max_len $max_len
done
    done
        done
            done
                done
                    done
