export CUDA_VISIBLE_DEVICES=1
python3 train.py \
  data=cluster \
  nn_params=cluster_encoder \
  train_params.batch_size=36 \
  train_params.num_iter=24000000 \
  train_params.num_iter_per_train_log=10 \
  train_params.num_iter_per_validation=30000 \
  train_params.num_iter_per_inference=30000 \
  train_params.num_iter_per_checkpoint=30000 \
  train_params.max_length=50 \
  inference_params.num_inference=5 \
  inference_params.sampling.method=argmax \
  general.log=true \
  general.infer=true