for i in 0 1 2 3 4 5 6 7 8 9 
do
	  python train.py --model_kind gpt --config config/gpt-$i.yml --model_desc gpt-lstm-feats-$i
done
