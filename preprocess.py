#!/usr/bin/python

import argparse #Arguments and flags
import os #List files in a directory
from mosestokenizer import * #Tokenize the text
import pickle #Saving preprocessed data
import sys #Max integer size

DEFAULT_SOURCE = "data/"
DEFAULT_DESTINATION = "data/"

"""
Cleans the input of any characters or strings that shouldn't be processed
@param input: the input that has to be cleaned
@return the cleaned input
"""
def getCleanLine(input):
	# @TODO
	return input.replace("\n", "")

"""
Processes and exports the commoncrawl data in a ready-to-use format
@param args: the argument parser
@param tokenizer: the moses tokenizer set to tokenize german
"""
def processCommoncrawl(args, tokenizer):
	# @TODO
	pass

"""
Processes and exports the europarl data in a ready-to-use format
@param args: the argument parser
@param tokenizer: the moses tokenizer set to tokenize german
"""
def processEuroparl(args, tokenizer):
	#Process German data
	inputFile = open(args.src + "Europarl/europarl-v7.de-en.de", "r")
	germanData = [] #List of samples in german
	i = 0 #For limiting the amount of samples
	for line in inputFile.readlines():
		if i >= args.cap:
			break
		else:
			i += 1
		germanData.append(tokenizer(getCleanLine(line)))
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
"""
def processParacrawl(args, tokenizer):
	# @TODO
	pass



def main(args):
	tokenizer = MosesTokenizer("de")
	"""with open(args.destination + OUTPUT_ID_KEY, "wb") as f:
		pickle.dump(ids, f, pickle.HIGHEST_PROTOCOL)"""
	if args.europarl:
		processEuroparl(args, tokenizer)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Preprocesses data for training a German-to-English machine translation system")
	#parser.add_argument('-t', dest='test', type=bool, default=False, help='This is a test argument, ignore it')
	parser.add_argument('-d', dest='dest', type=str, default=DEFAULT_DESTINATION, help="Destination directory for this script's output data")
	parser.add_argument('-s', dest='src', type=str, default=DEFAULT_SOURCE, help="Source directory for this script's input data")
	parser.add_argument('-n', dest='cap', type=int, default= sys.maxsize, help="Limits the amount of samples processed per source. Default=sys.maxsize")
	parser.add_argument('-e', action="store_true", dest='europarl', help="Processes the data from Europarl and exports it in europarl.pkl")
	parser.add_argument('-p', action="store_true", dest='paracrawl', help="Processes the data from ParaCrawl and exports it in paracrawl.pkl")
	args = parser.parse_args()
	main(args)