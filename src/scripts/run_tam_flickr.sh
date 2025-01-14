# nohup bash scripts/run_tam_flickr.sh 6 >logs/tam_flickr.log &
gpu=$1


for d in "Flickr"; do
    for cutting in 3 5 10; do
        for N_tree in 3; do
            for lamda in 0 0.5 1; do
                for readout in "max" "min" "avg" "weighted_sum"; do
                    for dim in 128 512 1024; do
                        python -u -m dgld.models.TAM.train --dataset $d --gpu $gpu  --cutting $cutting --N_tree $N_tree --lamda $lamda --embedding_dim $dim --readout $readout
done
    done
        done
            done
                done
                    done
