#!/usr/bin/python

import argparse #Arguments and flags
import os #List files in a directory
from mosestokenizer import * #Tokenize the text
import pickle #Saving preprocessed data
import sys #Max integer size
import sentencepiece as spm #For BPE

DEFAULT_SOURCE = "data/"
DEFAULT_DESTINATION = "data/"
DEFAULT_SPM_PATH = "m.model"
DEFAULT_SPM_TRAIN_DATA = "data/NewsCrawl/newscrawl.2019.de.shuffled.deduped" #@TODO change this to German when it's downloaded
DEFAULT_VOCAB_SIZE = 20000

"""
Segments the sample with the SentencePiece model
@param input: the input to be segmented
@param sp: the SentencePiece model
@return the segmented output
"""
def getSegLine(input, sp):
	result = []
	for word in input:
		for subword in sp.encode(word, out_type=str):
			result.append(subword)
	return result

"""
Loads a trained SentencePiece model saved as a binary pkl file.
If there is no SentencePiece model found, creates a new one, trains it, and saves it to disk.
@return: the SentencePiece model
"""
def getSentencePieceModel(args):
	if os.path.exists(args.spm):
		sp = spm.SentencePieceProcessor()
		sp.Load(args.spm)
		return sp
	else:
		if not os.path.exists(args.spt):
			print("ERROR: No training data for SentencePiece found at " + args.spt + "\nPlease rectify and run again")
			exit(1)
		spm.SentencePieceTrainer.train(input=args.spt, model_prefix="m", vocab_size=args.spv, model_type="unigram")
		sp = spm.SentencePieceProcessor()
		sp.Load(args.spm)
		return sp

"""
Cleans the input of any characters or strings that shouldn't be processed
@param input: the input that has to be cleaned
@return the cleaned input
"""
def getCleanLine(input):
	# @TODO
	return input.replace("\n", "").lower()

"""
Processes and exports the commoncrawl data in a ready-to-use format
@param args: the argument parser
@param tokenizer: the moses tokenizer set to tokenize german
@param sp: the SentencePiece model to BPE
"""
def processCommoncrawl(args, tokenizer, sp):
	# @TODO
	pass

"""
Processes and exports the europarl data in a ready-to-use format
@param args: the argument parser
@param tokenizer: the moses tokenizer set to tokenize german
@param sp: the SentencePiece model to BPE
"""
def processEuroparl(args, tokenizer, sp):
	#Process German data
	inputFile = open(args.src + "Europarl/europarl-v7.de-en.de", "r")
	germanData = [] #List of samples in german
	i = 0 #For limiting the amount of samples
	for line in inputFile.readlines():
		if i >= args.cap:
			break
		else:
			i += 1
		germanData.append(getSegLine(tokenizer(getCleanLine(line)), sp))
		print(germanData[i-1])
	inputFile.close()
	#Process English data
	inputFile = open(args.src + "Europarl/europarl-v7.de-en.en", "r")
	englishData = [] #List of samples in english
	i = 0
	for line in inputFile.readlines():
		if i >= args.cap:
			break
		else:
			i += 1
		englishData.append(line)
	inputFile.close()
	europarlData = (germanData, englishData)
	with open(args.dest + "europarlData.pkl", "wb") as f:
		pickle.dump(europarlData, f, pickle.HIGHEST_PROTOCOL)

"""
Processes and exports the paracrawl data in a ready-to-use format
@param args: the argument parser
@param tokenizer: the moses tokenizer set to tokenize german
@param sp: the SentencePiece model to BPE
"""
def processParacrawl(args, tokenizer, sp):
	# @TODO
	pass



def main(args):
	tokenizer = MosesTokenizer("de")
	"""with open(args.destination + OUTPUT_ID_KEY, "wb") as f:
		pickle.dump(ids, f, pickle.HIGHEST_PROTOCOL)"""
	sp = getSentencePieceModel(args)
	if args.europarl:
		processEuroparl(args, tokenizer, sp)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Preprocesses data for training a German-to-English machine translation system")
	#parser.add_argument('-t', dest='test', type=bool, default=False, help='This is a test argument, ignore it')
	parser.add_argument('-d', dest='dest', type=str, default=DEFAULT_DESTINATION, help="Destination directory for this script's output data. Default=" + DEFAULT_DESTINATION)
	parser.add_argument('-s', dest='src', type=str, default=DEFAULT_SOURCE, help="Source directory for this script's input data. Default=" + DEFAULT_SOURCE)
	parser.add_argument('-n', dest='cap', type=int, default= sys.maxsize, help="Limits the amount of samples processed per source. Default=sys.maxsize")
	parser.add_argument('-spm', dest='spm', type=str, default=DEFAULT_SPM_PATH, help="Sets the path to a trained SentencePiece model. Default=" + DEFAULT_SPM_PATH)
	parser.add_argument('-spt', dest='spt', type=str, default=DEFAULT_SPM_TRAIN_DATA, help="Sets the path to training data for a SentencePiece model. Default=" + DEFAULT_SPM_TRAIN_DATA)
	parser.add_argument('-spv', dest='spv', type=int, default=DEFAULT_VOCAB_SIZE, help="Sets the vocab size for the SentencePiece model. Default=" + str(DEFAULT_VOCAB_SIZE))
	parser.add_argument('-e', action="store_true", dest='europarl', help="Processes the data from Europarl and exports it in europarl.pkl")
	parser.add_argument('-p', action="store_true", dest='paracrawl', help="Processes the data from ParaCrawl and exports it in paracrawl.pkl")
	args = parser.parse_args()
	main(args)