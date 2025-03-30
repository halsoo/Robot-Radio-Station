export CUDA_VISIBLE_DEVICES=1
python3 inference.py \
  -w 2025-03-11-10-27-39:NaiveDecoderOnlyRecommender:dim-256:nLayers-4:dropout-0.2:batchSize-36:lrDecay-0.8:smp_naive \
  -d /home/dongmin/userdata/dongmin/robot-radio-station/metadata/smp_naive/test-segments-filtered.pt \
  --checkpoint best