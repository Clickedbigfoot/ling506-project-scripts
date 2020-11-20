global {
	ducttape_structure=flat
}

#Filters out bad sentence pairs and tokenizes the text
#Creates a trainingEn.txt trainingDe.txt, validation and testing versions of the former, as well as spmDataEn.txt and spmDataDe.txt in the data/ directory
task prepare_data < script=./preprocess.py spmTrainEn=data/NewsCrawl/newscrawl.2019.en.shuffled.deduped spmTrainDe=data/NewsCrawl/newscrawl.2019.de.shuffled.deduped source=data/ > trainEn=trainDataEn.txt trainDe=trainDataDe.txt testEn=testDataEn.txt testDe=testDataDe.txt valEn=valDataEn.txt valDe=valDataDe.txt spmEn=spmDataEn.txt spmDe=spmDataDe.txt :: N=10000 {
	python3 $script -n $N -s $source
	head -n $N < $spmTrainDe > $spmDe
	head -n $N < $spmTrainEn > $spmEn
}

task train_sentencepiece < script=./spTrainer.py inputEn=$spmEn@prepare_data inputDe=$spmDe@prepare_data > mEn=spEn.model mDe=spDe.model :: prefEn=./spEn prefDe=./spDe VS=32000 VSen=23000 userSym=<KN> {
	python3 $script --input $inputEn --model_prefix $prefEn --vocab_size $VSen --character_coverage 1 --user_defined_symbols $userSym
	python3 $script --input $inputDe --model_prefix $prefDe --vocab_size $VS --character_coverage 1 --user_defined_symbols $userSym
}

task run_sentencepiece_en < script=./spEncoder.py model=$mEn@train_sentencepiece trainI=$trainEn@prepare_data valI=$valEn@prepare_data testI=$testEn@prepare_data > train=trainDataEn.txt val=valDataEn.txt test=testDataEn.txt {
	spm_encode --model=$model --input $trainI --ouput $train
	spm_encode --model=$model --input $valI -output $val
	spm_encode --model=$model --input $testI --output $test
}

task run_sentencepiece_de < script=./spEncoder.py model=$mDe@train_sentencepiece trainI=$trainDe@prepare_data valI=$valDe@prepare_data testI=$testDe@prepare_data > train=trainDataDe.txt val=valDataDe.txt test=testDataDe.txt {
	python3 $script --model=$model --input $trainI --output $train
	python3 $script --model=$model --input $valI --output $val
	python3 $script --model=$model --input $testI --output $test
}