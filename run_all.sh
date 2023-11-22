for i in 0
do
	  python poetry_diacritizer/train.py --model_kind gpt --config config/gpt-$i.yml --model_desc gpt-cls-$i
done
