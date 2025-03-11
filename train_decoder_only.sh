python3 train.py \
  data=smp_naive \
  nn_params=decoder_only \
  train_params.batch_size=36 \
  train_params.num_iter=24000000 \
  train_params.num_iter_per_train_log=10 \
  train_params.num_iter_per_validation=30000 \
  train_params.num_iter_per_inference=300000 \
  train_params.num_iter_per_checkpoint=3000000 \
  train_params.max_length=30 \
  inference_params.num_inference=5 \
  inference_params.sampling.method=argmax \
  general.log=true \
  general.infer=true