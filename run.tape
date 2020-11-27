global {
	ducttape_structure=flat
}

#Filters out bad sentence pairs and tokenizes the text
#Creates a trainingEn.txt trainingDe.txt, validation and testing versions of the former, as well as spmDataEn.txt and spmDataDe.txt in the data/ directory
task prepare_data < script=./preprocess.py spmTrainEn=data/NewsCrawl/newscrawl.2019.en.shuffled.deduped spmTrainDe=data/NewsCrawl/newscrawl.2019.de.shuffled.deduped source=data/ > trainEn=trainDataEn.txt trainDe=trainDataDe.txt testEn=testDataEn.txt testDe=testDataDe.txt valEn=valDataEn.txt valDe=valDataDe.txt spmEn=spmDataEn.txt spmDe=spmDataDe.txt testR=testReplace.txt :: N=10000 {
	python3 $script -n $N -s $source
	head -n $N < $spmTrainDe > $spmDe
	head -n $N < $spmTrainEn > $spmEn
}

task train_sentencepiece < script=./spTrainer.py inputEn=$spmEn@prepare_data inputDe=$spmDe@prepare_data > mEn=spEn.model mDe=spDe.model :: prefEn=./spEn prefDe=./spDe VS=32000 VSen=23000 userSym=<KN> {
	python3 $script --input $inputEn --model_prefix $prefEn --vocab_size $VSen --character_coverage 1 --user_defined_symbols $userSym
	python3 $script --input $inputDe --model_prefix $prefDe --vocab_size $VS --character_coverage 1 --user_defined_symbols $userSym
}

task run_sentencepiece_en < script=./spEncoder.py model=$mEn@train_sentencepiece trainI=$trainEn@prepare_data valI=$valEn@prepare_data testI=$testEn@prepare_data > train=trainDataEn.txt val=valDataEn.txt test=testDataEn.txt {
	python3 $script --model=$model --input $trainI --output $train
	python3 $script --model=$model --input $valI --output $val
	python3 $script --model=$model --input $testI --output $test
}

task run_sentencepiece_de < script=./spEncoder.py model=$mDe@train_sentencepiece trainI=$trainDe@prepare_data valI=$valDe@prepare_data testI=$testDe@prepare_data > train=trainDataDe.txt val=valDataDe.txt test=testDataDe.txt {
	python3 $script --model=$model --input $trainI --output $train
	python3 $script --model=$model --input $valI --output $val
	python3 $script --model=$model --input $testI --output $test
}

task train_big_transformer < script=./configCreator.py template=./de_en.yaml trainDe=$train@run_sentencepiece_de trainEn=$train@run_sentencepiece_en trainSrc=$train@run_sentencepiece_de trainTgt=$train@run_sentencepiece_en valDataDe=$val@run_sentencepiece_de valDataEn=$val@run_sentencepiece_en > config=de_en0.yaml vSrc=system_data.vocab.src vTgt=system_data.vocab.tgt modelOutput1=foo1_step_1000.pt modelOutput2=foo2_step_1000.pt modelOutput3=foo3_step_1000.pt modelOutput4=foo4_step_1000.pt :: model1=foo1 model2=foo2 model3=foo3 model4=foo4 vc=system_data {
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainDe --train_path_tgt $trainEn --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $model1 --ini 1
	echo "" > $vSrc
	echo "" > $vTgt
	onmt_build_vocab -config $config
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
	onmt_train -config $config --master_port=8000 -world_size 4 -gpu_ranks 4 5 6
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainDe --train_path_tgt $trainEn --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $model2 --ini 2
	onmt_train -config $config --master_port=8000 -world_size 4 -gpu_ranks 4 5 6
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainDe --train_path_tgt $trainEn --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $model3 --ini 3
	onmt_train -config $config --master_port=8000 -world_size 4 -gpu_ranks 4 5 6
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainDe --train_path_tgt $trainEn --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $mode4 --ini 4
	onmt_train -config $config --master_port=8000 -world_size 4 -gpu_ranks 4 5 6
}

task score_models < decodeScript=./spDecoder.py spModel=$mEn@train_sentencepiece model1=$modelOutput1@train_big_transformer
model2=$modelOutput2@train_big_transformer model3=$modelOutput3@train_big_transformer model4=$modelOutput4@train_big_transformer src=$test@run_sentencepiece_de gold=$test@run_sentencepiece_en > pred=predictions.txt predDetok=predictionsDetokenized.txt {
	onmt_translate -model $model1 $model2 $model3 $model4 -src $src -output $pred
	python3 $decodeScript --model $spModel --input $pred --output $predDetok
	sacrebleu $gold < $predDetok
}
