# nohup bash scripts/run_CONAD.sh 3 >logs/CONAD_fb.log &
gpu=$1


# for d in "Facebook" "YelpChi"; do
for d in "Facebook"; do
    for eta in 0.01 0.1 0.5 0.8; do
        for lr in 0.01 0.001 0.0001; do
            for contrast_type in siamese triplet; do
                for alpha in 0.1 0.5 0.9; do
                    for margin in 0.1 0.5 0.9; do
                        python -u -m dgld.train.main --dataset $d --device $gpu --runs 3 CONAD --lr $lr --alpha $alpha --weight_decay 0.00001 --eta $eta --contrast_type $contrast_type --margin $margin
done
    done
        done
            done
                done
                    done
