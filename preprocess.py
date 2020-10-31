#!/usr/bin/python

import argparse #Arguments and flags
import os #List files in a directory
from mosestokenizer import * #Tokenize the text
import pickle #Saving preprocessed data

DEFAULT_SOURCE = "data/"
DEFAULT_DESTINATION = "data/"

"""
Processes and exports the europarl data in a ready-to-use format
@param args: the argument parser
"""
def processEuroparl(args):
	pass

"""
Processes and exports the paracrawl data in a ready-to-use format
@param args: the argument parser
"""
def processParacrawl(args):
	pass



def main(args):
	tokenizer = MosesTokenizer("en")
	"""with open(args.destination + OUTPUT_ID_KEY, "wb") as f:
		pickle.dump(ids, f, pickle.HIGHEST_PROTOCOL)"""

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Preprocesses data for training a German-to-English machine translation system")
	#parser.add_argument('-t', dest='test', type=bool, default=False, help='This is a test argument, ignore it')
	parser.add_argument('-d', dest='destination', type=str, default=DEFAULT_DESTINATION, help="Destination directory for this script's output data")
	parser.add_argument('-s', dest='source', type=str, default=DEFAULT_SOURCE, help="Source directory for this script's input data")
	parser.add_argument('-e', action="store_true", dest='europarl', help="Processes the data from Europarl and exports it in europarl.pkl")
	parser.add_argument('-p', action="store_true", dest='paracrawl', help="Processes the data from ParaCrawl and exports it in paracrawl.pkl")
	args = parser.parse_args()
	main(args)