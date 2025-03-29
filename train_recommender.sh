export CUDA_VISIBLE_DEVICES=0
python3 train.py \
  data=cluster \
  nn_params=cluster_encoder \
  nn_params.dim=512 \
  nn_params.depth=8 \
  train_params.batch_size=256 \
  train_params.num_iter=24000000 \
  train_params.num_iter_per_train_log=10 \
  train_params.num_iter_per_validation=3000 \
  train_params.num_iter_per_inference=30000 \
  train_params.num_iter_per_checkpoint=30000 \
  train_params.max_length=50 \
  inference_params.num_inference=100 \
  inference_params.sampling.method=argmax \
  inference_params.sampling.temperature=1.5 \
  general.log=true \
  general.infer=true