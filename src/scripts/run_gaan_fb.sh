# nohup bash scripts/run_gaan_fb.sh 0 >logs/gaan_fb.log &
gpu=$1


for d in "Facebook"; do
    for hid_dim in 256 128 64; do
        for lr in 0.001 0.0001; do
            for dropout in 0.2 0.5 0.8; do
                for num_layers in 2 3 5; do
                    for beta in 0.2 0.7; do
                        for noise_dim in 16 64; do
                                python -u -m dgld.models.pygod.train.gaan.gaan --dataset $d --gpu $gpu --hid_dim $hid_dim --lr $lr --dropout $dropout --num_layers $num_layers --beta $beta --noise_dim $noise_dim
done
    done
        done
            done
                done
                    done
                        done
