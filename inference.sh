export CUDA_VISIBLE_DEVICES=1
python3 inference.py \
  -w run-20250330_003517-jzan8ibe \
  -d /home/dongmin/userdata/dongmin/robot-radio-station/test_data/ \
  --checkpoint best