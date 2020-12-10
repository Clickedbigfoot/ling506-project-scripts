global {
	ducttape_structure=flat
}

#Filters out bad sentence pairs and tokenizes the text
#Creates a trainingEn.txt trainingDe.txt, validation and testing versions of the former, as well as spmDataEn.txt and spmDataDe.txt in the data/ directory
task prepare_data < script=./preprocess.py spmTrainEn=data/NewsCrawl/newscrawl.2019.en.shuffled.deduped spmTrainDe=data/NewsCrawl/newscrawl.2019.de.shuffled.deduped source=data/ > trainEn=trainDataEn.txt trainDe=trainDataDe.txt testEn=testDataEn.txt testDe=testDataDe.txt valEn=valDataEn.txt valDe=valDataDe.txt spmEn=spmDataEn.txt spmDe=spmDataDe.txt testR=testReplace.txt ftEn=ftEn.txt ftDe=ftDe.txt :: N=1000000 {
	python3 $script -n $N -s $source
	head -n $N < $spmTrainDe > $spmDe
	head -n $N < $spmTrainEn > $spmEn
}

task train_sentencepiece < script=./spTrainer.py inputEn=$spmEn@prepare_data inputDe=$spmDe@prepare_data > mEn=spEn.model mDe=spDe.model :: prefEn=./spEn prefDe=./spDe VS=32000 VSen=23000 userSym=<KN> {
	python3 $script --input $inputEn --model_prefix $prefEn --vocab_size $VSen --character_coverage 1 --user_defined_symbols $userSym
	python3 $script --input $inputDe --model_prefix $prefDe --vocab_size $VS --character_coverage 1 --user_defined_symbols $userSym
}

task run_sentencepiece_en < script=./spEncoder.py model=$mEn@train_sentencepiece trainI=$trainEn@prepare_data valI=$valEn@prepare_data testI=$testEn@prepare_data ftEnI=$ftEn@prepare_data > train=trainDataEn.txt val=valDataEn.txt test=testDataEn.txt ftEn=ftEn.txt {
	python3 $script --model=$model --input $trainI --output $train
	python3 $script --model=$model --input $valI --output $val
	python3 $script --model=$model --input $testI --output $test
	python3 $script --model=$model --input $ftEnI --output $ftEn
}

task run_sentencepiece_de < script=./spEncoder.py model=$mDe@train_sentencepiece trainI=$trainDe@prepare_data valI=$valDe@prepare_data testI=$testDe@prepare_data ftDeI=$ftDe@prepare_data > train=trainDataDe.txt val=valDataDe.txt test=testDataDe.txt ftDe=ftDe.txt {
	python3 $script --model=$model --input $trainI --output $train
	python3 $script --model=$model --input $valI --output $val
	python3 $script --model=$model --input $testI --output $test
	python3 $script --model=$model --input $ftDeI --output $ftDe
}

task baseline_model < script=./configCreator.py template=./de_en.yaml trainDe=$train@run_sentencepiece_de trainEn=$train@run_sentencepiece_en trainSrc=$train@run_sentencepiece_de trainTgt=$train@run_sentencepiece_en valDataDe=$val@run_sentencepiece_de valDataEn=$val@run_sentencepiece_en decodeScript=./spDecoder.py spModel=$mEn@train_sentencepiece testDe=$test@run_sentencepiece_de testEn=$test@run_sentencepiece_en > config=de_en0.yaml vSrc=system_data.vocab.src vTgt=system_data.vocab.tgt modelOutput1=foo1_step_4000000.pt pred=predictions.txt predDetok=predictionsDetokenized.txt :: model1=foo1 vc=system_data {
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainDe --train_path_tgt $trainEn --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $model1 --ini 1
	echo "" > $vSrc
	echo "" > $vTgt
	onmt_build_vocab -config $config
	CUDA_VISIBLE_DEVICES=4,5,6,7
	onmt_train -config $config --master_port=8000 -world_size 4 -gpu_ranks 0 1 2 3
	onmt_translate -model $model1 -src $testDe -output $pred
	python3 $decodeScript --model $spModel --input $pred --output $predDetok
	sacrebleu $testEn < $predDetok
}

task build_vocab < script=./configCreator.py template=./de_en.yaml trainDe=$train@run_sentencepiece_de trainEn=$train@run_sentencepiece_en valDataDe=$val@run_sentencepiece_de valDataEn=$val@run_sentencepiece_en trainFtDe=$ftDe@run_sentencepiece_de trainFtEn=$ftEn@run_sentencepiece_en > config=de_en0.yaml vSrc=system_data.vocab.src vTgt=system_data.vocab.tgt trainFE=trainFullEn.txt trainFD=trainFullDe.txt :: vc=system_data model=foo {
	#Build vocab
	cp $trainEn $trainFE
	cat $trainFtEn >> $trainFE
	cp $trainDe $trainFD
	cat $trainFtDe >> $trainFD
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainFD --train_path_tgt $trainFE --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $model --ini 0
	echo "" > $vSrc
	echo "" > $vTgt
	onmt_build_vocab -config $config
}

task ensemble1 < script=./configCreator.py template=./de_en.yaml trainDe=$train@run_sentencepiece_de trainEn=$train@run_sentencepiece_en valDataDe=$val@run_sentencepiece_de valDataEn=$val@run_sentencepiece_en trainFtDe=$ftDe@run_sentencepiece_de trainFtEn=$ftEn@run_sentencepiece_en vSrc=$vSrc@build_vocab vTgt=$vTgt@build_vocab > config=de_en0.yaml modelOutput=foo_step_4000000.pt modelOutputF=foo_step_16384.pt :: vc=system_data model=foo {
	#Train first model
	CUDA_VISIBLE_DEVICES=4,5,6,7
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainDe --train_path_tgt $trainEn --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $model --ini 1
	onmt_train -config $config --master_port=8000 -world_size 4 -gpu_ranks 0 1 2 3
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainFtDe --train_path_tgt $trainFtEn --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $model --ini 1
	onmt_train -config $config --master_port=8000 -world_size 4 -gpu_ranks 0 1 2 3 --train_steps 16384 --save_checkpoint_steps 16384 --train_from $modelOutput --learning_rate 0.1
}

task ensemble2 < script=./configCreator.py template=./de_en.yaml trainDe=$train@run_sentencepiece_de trainEn=$train@run_sentencepiece_en valDataDe=$val@run_sentencepiece_de valDataEn=$val@run_sentencepiece_en trainFtDe=$ftDe@run_sentencepiece_de trainFtEn=$ftEn@run_sentencepiece_en vSrc=$vSrc@build_vocab vTgt=$vTgt@build_vocab > config=de_en0.yaml modelOutput=foo_step_4000000.pt modelOutputF=foo_step_16384.pt :: vc=system_data model=foo {
	#Train second model
	CUDA_VISIBLE_DEVICES=4,5,6,7
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainDe --train_path_tgt $trainEn --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $model --ini 2
	onmt_train -config $config --master_port=8000 -world_size 4 -gpu_ranks 0 1 2 3
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainFtDe --train_path_tgt $trainFtEn --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $model --ini 2
	onmt_train -config $config --master_port=8000 -world_size 4 -gpu_ranks 0 1 2 3 --train_steps 16384 --save_checkpoint_steps 16384 --train_from $modelOutput --learning_rate 0.1
}

task ensemble3 < script=./configCreator.py template=./de_en.yaml trainDe=$train@run_sentencepiece_de trainEn=$train@run_sentencepiece_en valDataDe=$val@run_sentencepiece_de valDataEn=$val@run_sentencepiece_en trainFtDe=$ftDe@run_sentencepiece_de trainFtEn=$ftEn@run_sentencepiece_en vSrc=$vSrc@build_vocab vTgt=$vTgt@build_vocab > config=de_en0.yaml modelOutput=foo_step_4000000.pt modelOutputF=foo_step_16384.pt :: vc=system_data model=foo {
	#Train third model
	CUDA_VISIBLE_DEVICES=4,5,6,7
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainDe --train_path_tgt $trainEn --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $model --ini 3
	onmt_train -config $config --master_port=8000 -world_size 4 -gpu_ranks 0 1 2 3
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainFtDe --train_path_tgt $trainFtEn --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $model --ini 3
	onmt_train -config $config --master_port=8000 -world_size 4 -gpu_ranks 0 1 2 3 --train_steps 16384 --save_checkpoint_steps 16384 --train_from $modelOutput --learning_rate 0.1
}

task ensemble4 < script=./configCreator.py template=./de_en.yaml trainDe=$train@run_sentencepiece_de trainEn=$train@run_sentencepiece_en valDataDe=$val@run_sentencepiece_de valDataEn=$val@run_sentencepiece_en trainFtDe=$ftDe@run_sentencepiece_de trainFtEn=$ftEn@run_sentencepiece_en vSrc=$vSrc@build_vocab vTgt=$vTgt@build_vocab > config=de_en0.yaml modelOutput=foo_step_4000000.pt modelOutputF=foo_step_16384.pt :: vc=system_data model=foo {
	#Train fourth model
	CUDA_VISIBLE_DEVICES=4,5,6,7
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainDe --train_path_tgt $trainEn --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $model --ini 4
	onmt_train -config $config --master_port=8000 -world_size 4 -gpu_ranks 0 1 2 3
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainFtDe --train_path_tgt $trainFtEn --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $model --ini 4
	onmt_train -config $config --master_port=8000 -world_size 4 -gpu_ranks 0 1 2 3 --train_steps 16384 --save_checkpoint_steps 16384 --train_from $modelOutput --learning_rate 0.1
}

task score_ensemble < decodeScript=./spDecoder.py spModel=$mEn@train_sentencepiece testDe=$test@run_sentencepiece_de testEn=$test@run_sentencepiece_en model1=$modelOutputF@ensemble1 model2=$modelOutputF@ensemble2 model3=$modelOutputF@ensemble3 model4=$modelOutputF@ensemble4 > pred=predictions.txt predDetok=predictionsDetok.txt {
	#Score ensemble
	onmt_translate -model $modelOutputF1 $modelOutputF2 $modelOutputF3 $modelOutputF4 -src $testDe -output $pred
	python3 $decodeScript --model $spModel --input $pred --output $predDetok
	sacrebleu $testEn < $predDetok
}