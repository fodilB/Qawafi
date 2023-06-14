for i in 0 1 2 3 4 5 6 7 8 9
do
	  python train.py --model_kind gpt --config config/gpt-cls-$i-tash-proc.yml --model_desc gpt-cls-feats-$i-tash-proc
done