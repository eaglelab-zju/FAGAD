# nohup bash scripts/run_tam_yelp.sh 5 >logs/tam_yelp_0.001.log &
gpu=$1


for d in "YelpChi"; do
    for cutting in 3 5 10; do
        for lr in 0.00001; do
        # for lr in 0.001; do
            for N_tree in 3; do
                for lamda in 0 0.5 1; do
                    for readout in "max" "avg" "weighted_sum" "min"; do
                        for dim in 512 128; do
                        # 1024 OOM
                            python -u -m dgld.models.TAM.train_large_graph --dataset $d --gpu $gpu  --cutting $cutting --N_tree $N_tree --lamda $lamda --embedding_dim $dim --readout $readout --lr $lr
done
    done
        done
            done
                done
                    done
                        done
