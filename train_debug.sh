python3 train.py \
  data=smp_naive_debug \
  nn_params=decoder_only \
  train_params.batch_size=36 \
  train_params.num_iter=1000 \
  train_params.num_iter_per_train_log=10 \
  train_params.num_iter_per_validation=500 \
  train_params.num_iter_per_inference=1000 \
  train_params.num_iter_per_checkpoint=1000 \
  train_params.max_length=30 \
  inference_params.num_inference=3 \
  inference_params.sampling.method=argmax \
  general.debug=true \
  general.log=false \
  general.infer=true