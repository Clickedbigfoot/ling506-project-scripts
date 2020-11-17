#!/usr/bin/python

import argparse #Arguments and flags
import os #List files in a directory
from mosestokenizer import * #Tokenize the text
import pickle #Saving preprocessed data
import sys #Max integer size
import sentencepiece as spm #For BPE
from langdetect import detect #For determining the language of the sample
from langdetect import DetectorFactory #Seed the langdetect module for consistency
import re #Regular expression usage

DEFAULT_SOURCE = "data/"
DEFAULT_DESTINATION = "data/"
DEFAULT_SPM_PATH = "m.model"
DEFAULT_SPM_TRAIN_DATA = "data/NewsCrawl/newscrawl.2019.de.shuffled.deduped.short"
DEFAULT_VOCAB_SIZE = 50000
SHORT = ".short" #Change to ".short" to load short versions of data or keep empty string to load full data

"""
Determines whether or not the sentence has too few words, too many, or any words that are too long
@param sentence: the sentence being examined
"""
def isBadSize(sentence):
	nWords = 0 #Amount of words in the sentence
	for token in sentence:
		#Iterate through every token in the sentence after it was tokenized
		if token[0:1] in string.punctuation:
			#This is punctuation
			continue
		if len(token) > 40:
			return True
		nWords += 1
	if nWords < 4 or 100 < nWords:
		return True
	return False

"""
Filters out samples that are detrimental to the system's training and tokenizes the input.
A sample is filtered out for the following reasons:
1) It is not German on the input side and English on the output side
2) A word is longer than 40 characters
3) The character ratio between a pair is greater than 1:3 or 3:1
4) There are less than four words in one sentence
5) There are greater than four words in one sentence
Any eligible sample has the German side tokenized with Moses tokenizer
@param german: the german sentence being examined
@param english: the english sentence being examined
@param tokenizer: the moses tokenizer
@param tokenizerEn: the moses tokenizer for english
@return a tuple of (germanLine, englishLine) if the sample is usable, (None, None) otherwise
"""
def getFiltered(german, english, tokenizer, tokenizerEn):
	if detect(german) != "de" or detect(english) != "en":
		return (None, None)
	lenEn = len(english)
	lenDe = len(german)
	if lenEn < lenDe and lenEn * 3 < lenDe:
		return (None, None)
	if lenDe < lenEn and lenDe * 3 < lenEn:
		return (None, None)
	germanTok = tokenizer(german)
	if isBadSize(germanTok) or isBadSize(tokenizerEn(english)):
		return (None, None)
	return (germanTok, english)


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
	# @TODO Filter out reddit usernames, emails, or html
	result = "".join([c for c in input if c.isprintable()])
	return result.replace("\n", "").lower()

"""
Processes and exports the commoncrawl data in a ready-to-use format
@param args: the argument parser
@param tokenizer: the moses tokenizer set to tokenize german
@param sp: the SentencePiece model to BPE
"""
def processCommoncrawl(args, tokenizer, sp):
	germanData = []
	englishData = []
	tokenizerEn = MosesTokenizer("en")
	inputFile = open(args.src + "CommonCrawl/commoncrawl.de-en.de" + SHORT, "r")
	i = 0
	for line in inputFile.readlines():
		if i >= args.cap:
			break;
		else:
			i += 1
		germanData.append(getCleanLine(line))
	inputFile.close()
	inputFile = open(args.src + "CommonCrawl/commoncrawl.de-en.en" + SHORT, "r")
	i = 0
	for line in inputFile.readlines():
		if i >= args.cap:
			break;
		else:
			i += 1
		englishData.append(getCleanLine(line))
	inputFile.close()
	#Now filter them
	germanDataF = []
	englishDataF = []
	for i in range(0, len(germanData)):
		(germanLine, englishLine) = getFiltered(germanData[i], englishData[i], tokenizer, tokenizerEn)
		if germanLine == None or englishLine == None:
			continue
		germanDataF.append(germanLine)
		englishDataF.append(englishLine)
	commoncrawlData = (germanDataF, englishDataF)
	with open(args.dest + "commoncrawlData.pkl", "wb") as f:
		pickle.dump(commoncrawlData, f, pickle.HIGHEST_PROTOCOL)

"""
Processes and exports the europarl data in a ready-to-use format
@param args: the argument parser
@param tokenizer: the moses tokenizer set to tokenize german
@param sp: the SentencePiece model to BPE
"""
def processEuroparl(args, tokenizer, sp):
	#Process German data
	inputFile = open(args.src + "Europarl/europarl-v7.de-en.de" + SHORT, "r")
	germanData = [] #List of samples in german
	i = 0 #For limiting the amount of samples
	for line in inputFile.readlines():
		if i >= args.cap:
			break
		else:
			i += 1
		germanData.append(getSegLine(tokenizer(getCleanLine(line)), sp))
	inputFile.close()
	#Process English data
	inputFile = open(args.src + "Europarl/europarl-v7.de-en.en" + SHORT, "r")
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
	germanData = []
	englishData = []
	tokenizerEn = MosesTokenizer("en")
	inputFile = open(args.src + "ParaCrawl/en-de.txt" + SHORT, "r")
	i = 0
	for line in inputFile.readlines():
		if i >= args.cap:
			break
		else:
			i += 1
		germanLine = getCleanLine(line[:line.index("\t")])
		englishLine = getCleanLine(line[line.index("\t")+1:])
		(germanLine, englishLine) = getFiltered(germanLine, englishLine, tokenizer, tokenizerEn)
		if germanLine == None or englishLine == None:
			continue
		germanData.append(germanLine)
		englishData.append(englishLine)
	inputFile.close()
	paracrawlData = (germanData, englishData)
	with open(args.dest + "paracrawlData.pkl", "wb") as f:
		pickle.dump(paracrawlData, f, pickle.HIGHEST_PROTOCOL)


def main(args):
	tokenizer = MosesTokenizer("de")
	DetectorFactory.seed = 0
	sp = getSentencePieceModel(args)
	if args.europarl:
		processEuroparl(args, tokenizer, sp)
	if args.paracrawl:
		processParacrawl(args, tokenizer, sp)

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

"""with open(args.destination + OUTPUT_ID_KEY, "wb") as f:
		pickle.dump(ids, f, pickle.HIGHEST_PROTOCOL)"""