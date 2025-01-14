nohup nnictl create --config dgld/models/pygod/train/gadnr/gadnr_flickr.yaml --port 8885 > logs/gadnr_flickr.log &
# sgrunawz 54nlox9k
# 256 OOM
nohup nnictl create --config dgld/models/pygod/train/gadnr/gadnr_ama.yaml --port 8888 > logs/gadnr_ama.log &
# ijrcd6z9
nohup nnictl create --config dgld/models/pygod/train/gadnr/gadnr_fb.yaml --port 8887 > logs/gadnr_fb.log &
# jw3c8br1 vz6bdipa  0o8rdxgl


nohup nnictl create --config dgld/models/pygod/train/gadnr/gadnr_blog.yaml --port 8882 > logs/gadnr_blog.log &
nohup nnictl create --config dgld/models/pygod/train/gadnr/gadnr_red.yaml --port 8884 > logs/gadnr_red.log &
nohup nnictl create --config dgld/models/pygod/train/gadnr/gadnr_wiki.yaml --port 8885 > logs/gadnr_wiki.log &
nohup nnictl create --config dgld/models/pygod/train/gadnr/gadnr_enr.yaml --port 8886 > logs/gadnr_enr.log &
nohup nnictl create --config dgld/models/pygod/train/gadnr/gadnr_yelp.yaml --port 8889 > logs/gadnr_yelp.log &
