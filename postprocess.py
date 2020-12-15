#!/usr/bin/python

import argparse #Arguments and flags
import os #List files in a directory
from mosestokenizer import * #Tokenize the text
import sys #Max integer size
import re #Regular expression usage
import string #For getting a set of punctuation

REPLACE_TOKEN = "<KN>"
DELIMITER = ";$;"

"""
Makes the necessary replacements and returns the complete line
@param predictions: string of the predictions made by the model
@param replacements: string of the replacements to insert
@return string of the predictions line with the replacements made
"""
def makeReplacements(predictions, replacements):
	outputLine = predictions
	replL = replacements.split(DELIMITER) #Last item on the list is, at least, newline byte
	while len(replL > 1):
		#As long as there is more than the newline byte
		if REPLACE_TOKEN not in outputLine:
			break
		outputLine = outputLine.replace(REPLACE_TOKEN, replL[0], 1)
		replL = replL[1:]
	return outputLine

def main(args):
	if args.input == "" or args.output == "" or args.repl == "":
		print("Invalid usage. Run with -h flag for details.")
		exit(1)
	inputFile = open(args.input, "r")
	outputFile = open(args.output, "w+")
	replaceFile = open(args.repl, "r")
	while 1:
		prediction = inputFile.readline()
		if prediction == "":
			break #EOF reached
		replacements = replaceFile.readline()
		prediction = prediction.replace("«", '"')
		prediction = prediction.replace("»", '"')
		prediction = makeReplacements(prediction, replacements)
		outputFile.write(prediction)
	inputFile.close()
	outputFile.close()
	replaceFile.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Preprocesses data for training a German-to-English machine translation system")
	#parser.add_argument('-t', dest='test', type=bool, default=False, help='This is a test argument, ignore it')
	parser.add_argument('-i', dest='input', type=str, default="", help="The file to be postprocessed")
	parser.add_argument('-o', dest='output', type=str, default="", help="The output of postprocessed translations")
	parser.add_argument('-r', dest='repl', type=str, default="", help="The file with the replacements to insert")
	args = parser.parse_args()
	main(args)

"""with open(args.destination + OUTPUT_ID_KEY, "wb") as f:
		pickle.dump(ids, f, pickle.HIGHEST_PROTOCOL)"""