global {
	ducttape_structure=flat
	nFt=42000
}

task download_data > europarlDe=europarl-v7.de-en.de europarlEn=europarl-v7.de-en.en paracrawl=en-de.txt commoncrawlDe=commoncrawl.de-en.de commoncrawlEn=commoncrawl.de-en.en newscrawlEn=news.2019.en.shuffled.deduped newscrawlDe=news.2019.de.shuffled.deduped {
	#Download Europarl
	wget http://www.statmt.org/europarl/v7/de-en.tgz
	tar -xzvf de-en.tgz
	#Download Paracrawl
	wget https://s3.amazonaws.com/web-language-models/paracrawl/release5.1/en-de.txt.gz
	gzip -dk en-de.txt.gz
	#Download Commoncrawl
	wget statmt.org/wmt13/training-parallel-commoncrawl.tgz
	tar -xzvf training-parallel-commoncrawl.tgz
	#Download newscrawl
	wget data.statmt.org/news-crawl/en/news.2019.en.shuffled.deduped.gz
	gzip -dk news.2019.en.shuffled.deduped.gz
	wget data.statmt.org/news-crawl/de/news.2019.de.shuffled.deduped.gz
	gzip -dk news.2019.de.shuffled.deduped.gz
}

task backtranslate < newscrawl=$newscrawlEn@download_data encodeScript=./spEncoder.py decodeScript=./spDecoder.py > de=predDetok.txt en=en.txt :: N=15000 {
	#Shorten newscrawl data
	head -n $N < $newscrawl | sed '/^[[:space:]]*$/d' | sed '/^.\{200\}./d' > $en
	#Download pretrained models
	wget https://s3.amazonaws.com/opennmt-models/transformer-ende-wmt-pyOnmt.tar.gz
	tar -xzvf transformer-ende-wmt-pyOnmt.tar.gz
	#Backtranslate
	python3 $encodeScript --model sentencepiece.model --input $en --output enTok.txt -s
	onmt_translate -model averaged-10-epoch.pt -src enTok.txt --output pred.txt
	python3 $decodeScript --model sentencepiece.model --input pred.txt --output predDetok.txt -s
}

#Filters out bad sentence pairs and tokenizes the text
#Creates a trainingEn.txt trainingDe.txt, validation and testing versions of the former, as well as spmDataEn.txt and spmDataDe.txt in the data/ directory
task prepare_data < script=./preprocess.py spmTrainEn=$newscrawlEn@download_data spmTrainDe=$newscrawlDe@download_data btDe=$de@backtranslate btEn=$en@backtranslate euroDe=$europarlDe@download_data euroEn=$europarlEn@download_data ccDe=$commoncrawlDe@download_data ccEn=$commoncrawlEn@download_data pc=$paracrawl@download_data > trainEn=trainDataEn.txt trainDe=trainDataDe.txt testEn=testDataEn.txt testDe=testDataDe.txt valEn=valDataEn.txt valDe=valDataDe.txt spmEn=spmDataEn.txt spmDe=spmDataDe.txt testR=testReplace.txt ftEn=ftEn.txt ftDe=ftDe.txt :: N=10000 {
	python3 $script -n $N --backtranslations_de $btDe --backtranslations_en $btEn --europarl_de $euroDe --europarl_en $euroEn --commoncrawl_de $ccDe --commoncrawl_en $ccEn --paracrawl $pc
	head -n 500000 < $spmTrainDe > $spmDe
	head -n 500000 < $spmTrainEn > $spmEn
}

#Original vocab size was 23000 for training sizes of 1000000
task train_sentencepiece < script=./spTrainer.py inputEn=$spmEn@prepare_data inputDe=$spmDe@prepare_data > mEn=spEn.model mDe=spDe.model :: prefEn=./spEn prefDe=./spDe VS=23000 VSen=23000 userSym=<KN> {
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

task baseline_model < decodeScript=./spDecoder.py spModel=$mEn@train_sentencepiece testDe=$test@run_sentencepiece_de testEn=$testEn@prepare_data model1=$modelOutput@ensemble1 postScript=./postprocess.py testR=$testR@prepare_data > pred=predictions.txt predDetok=predictionsDetokenized.txt predPost=predictionsPost.txt {
	export CUDA_VISIBLE_DEVICES=6,7
	onmt_translate -model $model1 -src $testDe -output $pred
	python3 $decodeScript --model $spModel --input $pred --output $predDetok
	python3 $postScript -i $predDetok -o $predPost -r $testR
	sacrebleu $testEn < $predPost
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

task ensemble1 < script=./configCreator.py template=./de_en.yaml trainDe=$train@run_sentencepiece_de trainEn=$train@run_sentencepiece_en valDataDe=$val@run_sentencepiece_de valDataEn=$val@run_sentencepiece_en trainFtDe=$ftDe@run_sentencepiece_de trainFtEn=$ftEn@run_sentencepiece_en vSrc=$vSrc@build_vocab vTgt=$vTgt@build_vocab > config=de_en0.yaml modelOutput=foo_step_40000.pt modelOutputF=foo_step_42000.pt :: vc=system_data model=foo ftSteps=$nFt {
	#Train first model
	export CUDA_VISIBLE_DEVICES=4,5,6,7
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainDe --train_path_tgt $trainEn --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $model --ini 1
	onmt_train -config $config --master_port=8000 -world_size 4 -gpu_ranks 0 1 2 3
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainFtDe --train_path_tgt $trainFtEn --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $model --ini 1
	onmt_train -config $config --master_port=8000 -world_size 4 -gpu_ranks 0 1 2 3 --train_steps $ftSteps --save_checkpoint_steps 1000 --train_from $modelOutput --learning_rate 0.1
}

task ensemble2 < script=./configCreator.py template=./de_en.yaml trainDe=$train@run_sentencepiece_de trainEn=$train@run_sentencepiece_en valDataDe=$val@run_sentencepiece_de valDataEn=$val@run_sentencepiece_en trainFtDe=$ftDe@run_sentencepiece_de trainFtEn=$ftEn@run_sentencepiece_en vSrc=$vSrc@build_vocab vTgt=$vTgt@build_vocab > config=de_en0.yaml modelOutput=foo_step_40000.pt modelOutputF=foo_step_42000.pt :: vc=system_data model=foo ftSteps=$nFt {
	#Train first model
	export CUDA_VISIBLE_DEVICES=4,5,6,7
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainDe --train_path_tgt $trainEn --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $model --ini 1
	onmt_train -config $config --master_port=8000 -world_size 4 -gpu_ranks 0 1 2 3
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainFtDe --train_path_tgt $trainFtEn --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $model --ini 1
	onmt_train -config $config --master_port=8000 -world_size 4 -gpu_ranks 0 1 2 3 --train_steps $ftSteps --save_checkpoint_steps 1000 --train_from $modelOutput --learning_rate 0.1
}

task ensemble3 < script=./configCreator.py template=./de_en.yaml trainDe=$train@run_sentencepiece_de trainEn=$train@run_sentencepiece_en valDataDe=$val@run_sentencepiece_de valDataEn=$val@run_sentencepiece_en trainFtDe=$ftDe@run_sentencepiece_de trainFtEn=$ftEn@run_sentencepiece_en vSrc=$vSrc@build_vocab vTgt=$vTgt@build_vocab > config=de_en0.yaml modelOutput=foo_step_40000.pt modelOutputF=foo_step_42000.pt :: vc=system_data model=foo ftSteps=$nFt {
	#Train first model
	export CUDA_VISIBLE_DEVICES=4,5,6,7
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainDe --train_path_tgt $trainEn --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $model --ini 1
	onmt_train -config $config --master_port=8000 -world_size 4 -gpu_ranks 0 1 2 3
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainFtDe --train_path_tgt $trainFtEn --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $model --ini 1
	onmt_train -config $config --master_port=8000 -world_size 4 -gpu_ranks 0 1 2 3 --train_steps $ftSteps --save_checkpoint_steps 1000 --train_from $modelOutput --learning_rate 0.1
}

task ensemble4 < script=./configCreator.py template=./de_en.yaml trainDe=$train@run_sentencepiece_de trainEn=$train@run_sentencepiece_en valDataDe=$val@run_sentencepiece_de valDataEn=$val@run_sentencepiece_en trainFtDe=$ftDe@run_sentencepiece_de trainFtEn=$ftEn@run_sentencepiece_en vSrc=$vSrc@build_vocab vTgt=$vTgt@build_vocab > config=de_en0.yaml modelOutput=foo_step_40000.pt modelOutputF=foo_step_42000.pt :: vc=system_data model=foo ftSteps=$nFt {
	#Train first model
	export CUDA_VISIBLE_DEVICES=4,5,6,7
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainDe --train_path_tgt $trainEn --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $model --ini 1
	onmt_train -config $config --master_port=8000 -world_size 4 -gpu_ranks 0 1 2 3
	python3 $script -t $template -o $config --src_vocab $vSrc --tgt_vocab $vTgt --save_data $vc --train_path_src $trainFtDe --train_path_tgt $trainFtEn --val_path_src $valDataDe --val_path_tgt $valDataEn --save_model $model --ini 1
	onmt_train -config $config --master_port=8000 -world_size 4 -gpu_ranks 0 1 2 3 --train_steps $ftSteps --save_checkpoint_steps 1000 --train_from $modelOutput --learning_rate 0.1
}

task score_ensemble < decodeScript=./spDecoder.py spModel=$mEn@train_sentencepiece testDe=$test@run_sentencepiece_de testEn=$testEn@prepare_data model1=$modelOutputF@ensemble1 model2=$modelOutputF@ensemble2 model3=$modelOutputF@ensemble3 model4=$modelOutputF@ensemble4 postScript=./postprocess.py testR=$testR@prepare_data > pred=predictions.txt predDetok=predictionsDetok.txt predPost=predictionsPost.txt {
	#Score ensemble
	onmt_translate -model $model1 $model2 $model3 $model4 -src $testDe -output $pred --gpu 4
	python3 $decodeScript --model $spModel --input $pred --output $predDetok
	python3 $postScript -i $predDetok -o $predPost -r $testR
	sacrebleu $testEn < $predPost
}