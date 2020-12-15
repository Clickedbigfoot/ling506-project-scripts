#!/usr/bin/python

import argparse #Arguments and flags
import os #List files in a directory
from mosestokenizer import * #Tokenize the text
import sys #Max integer size
from langdetect import detect #For determining the language of the sample
from langdetect import DetectorFactory #Seed the langdetect module for consistency
import re #Regular expression usage
import string #For getting a set of punctuation
import emoji #For extrscting emojis

TRAIN_DATA_EN = "trainDataEn.txt"
TRAIN_DATA_DE = "trainDataDe.txt"
VAL_DATA_EN = "valDataEn.txt"
VAL_DATA_DE = "valDataDe.txt"
TEST_DATA_EN = "testDataEn.txt"
TEST_DATA_DE = "testDataDe.txt"
TEST_DATA_REPLACE = "testReplace.txt"
TRAIN_FT_DE = "ftDe.txt"
TRAIN_FT_EN = "ftEn.txt"
EMAIL_PATTERN = r"[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[\.]\w+[\.]?\w+" #For emails
HTML_PATTERN = "<.*?>"
REPLACE_TOKEN = "<KN>"
DELIMITER = ";$;"

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
Filters out samples that are detrimental to the system's training
A sample is filtered out for the following reasons:
1) It is not German on the input side and English on the output side
2) A word is longer than 40 characters
3) The character ratio between a pair is greater than 1:3 or 3:1
4) There are less than four words in one sentence
5) There are greater than four words in one sentence
@param german: the german sentence being examined
@param english: the english sentence being examined
@param tokenizer: dict of moses tokenizers
@return a tuple of (germanLine, englishLine) if the sample is usable, (None, None) otherwise
"""
def getFiltered(german, english, tokenizer):
	try:
		germanD = detect(german)
		englishD = detect(english)
	except:
		#Could not detect any language
		return (None, None)
	if germanD != "de" or englishD != "en":
		return (None, None)
	lenEn = len(english)
	lenDe = len(german)
	if lenEn < lenDe and lenEn * 3 < lenDe:
		return (None, None)
	if lenDe < lenEn and lenDe * 3 < lenEn:
		return (None, None)
	germanTok = tokenizer["de"](german)
	englishTok = tokenizer["en"](english)
	if isBadSize(germanTok) or isBadSize(englishTok):
		return (None, None)
	return (german, english)


"""
Cleans the input of any characters or strings that shouldn't be processed
@param input: the input that has to be cleaned
@return the cleaned input
"""
def getCleanLine(input):
	# @TODO Filter out reddit usernames, emails, or html
	result = "".join([c for c in input if c.isprintable()])
	offenders = re.findall(HTML_PATTERN, result) #Get rid of errant html
	while len(offenders) > 0:
		result = result.replace(offenders[0], "")
		offenders = re.findall(HTML_PATTERN, result)
	offenders = re.findall(EMAIL_PATTERN, result)
	while len(offenders) > 0:
		result = result.replace(offenders[0], REPLACE_TOKEN)
		offenders = re.findall(EMAIL_PATTERN, result)
	scannee = result
	result = ""
	for c in scannee:
		if c not in emoji.UNICODE_EMOJI:
			result = result + c
	return result.replace("\n", "").lower()

"""
Cleans the input of any characters or strings that shouldn't be processed
@param input: the input that has to be cleaned
@return the cleaned input and a string of tokens that have been replaced in the format (cleanedLine, "replacements")
"""
def getCleanLineR(input):
	# @TODO Filter out reddit usernames, emails, or html
	result = "".join([c for c in input if c.isprintable()])
	replacements = ""
	offenders = re.findall(HTML_PATTERN, result) #Get rid of errant html
	while len(offenders) > 0:
		result = result.replace(offenders[0], "")
		offenders = re.findall(HTML_PATTERN, result)
	offenders = re.findall(EMAIL_PATTERN, result)
	while len(offenders) > 0:
		result = result.replace(offenders[0], REPLACE_TOKEN)
		replacements = replacements + offenders[0] + DELIMITER
		offenders = re.findall(EMAIL_PATTERN, result)
	scannee = result
	result = ""
	for c in scannee:
		if c in emoji.UNICODE_EMOJI:
			replacements = replacements + c + DELIMITER
		else:
			result = result + c
	return (result.replace("\n", "").lower(), replacements)

"""
Processes and exports the backtranslations data in a ready-to-use format
@param args: the argument parser
@param fds: dict of the files to which data will be written
@param tokenizer: dict of the tokenizers
"""
def processBacktranslations(args, fds, tokenizer):
	print("Processing data from Backtranslations")
	germanData = []
	englishData = []
	inputFile = open(args.backtranslationsDe, "r")
	inputFileEn = open(args.backtranslationsEn, "r")
	i = 0
	k = args.cap + round(args.cap * 0.01) #Cap for validation set
	j = k + round(args.cap * 0.01) #Cap for testing set
	while 1:
		line = inputFile.readline()
		if line == "":
			#EOF reached
			break
		(germanLine, replacements) = getCleanLineR(line)
		englishLine = getCleanLine(inputFileEn.readline())
		(germanLine, englishLine) = getFiltered(germanLine, englishLine, tokenizer)
		if germanLine == None or englishLine == None:
			continue
		if i < args.cap:
			fds["trainDe"].write(germanLine + "\n")
			fds["trainEn"].write(englishLine + "\n")
		elif i < k:
			fds["valDe"].write(germanLine + "\n")
			fds["valEn"].write(englishLine + "\n")
		else:
			fds["testDe"].write(germanLine + "\n")
			fds["testEn"].write(englishLine + "\n")
			fds["testR"].write(replacements + "\n")
		i += 1 #Iterate forward
		if i >= j:
			#Finished with the testing set as well
			break
	inputFile.close()
	inputFileEn.close()
	print("Saved " + str(args.cap) + " sentence pairs from Backtranslations to training data")

"""
Processes and exports the commoncrawl data in a ready-to-use format
@param args: the argument parser
@param fds: dict of the files to which data will be written
@param tokenizer: dict of the tokenizers
"""
def processCommoncrawl(args, fds, tokenizer):
	print("Processing data from CommonCrawl")
	germanData = []
	englishData = []
	inputFile = open(args.commoncrawlDe, "r")
	inputFileEn = open(args.commoncrawlEn, "r")
	i = 0
	k = args.cap + round(args.cap * 0.01) #Cap for validation set
	j = k + round(args.cap * 0.01) #Cap for testing set
	while 1:
		line = inputFile.readline()
		if line == "":
			#EOF reached
			break
		(germanLine, replacements) = getCleanLineR(line)
		englishLine = getCleanLine(inputFileEn.readline())
		(germanLine, englishLine) = getFiltered(germanLine, englishLine, tokenizer)
		if germanLine == None or englishLine == None:
			continue
		if i < args.cap:
			fds["trainDe"].write(germanLine + "\n")
			fds["trainEn"].write(englishLine + "\n")
		elif i < k:
			fds["valDe"].write(germanLine + "\n")
			fds["valEn"].write(englishLine + "\n")
		else:
			fds["testDe"].write(germanLine + "\n")
			fds["testEn"].write(englishLine + "\n")
			fds["testR"].write(replacements + "\n")
		i += 1 #Iterate forward
		if i >= j:
			#Finished with the testing set as well
			break
	inputFile.close()
	inputFileEn.close()
	print("Saved " + str(args.cap) + " sentence pairs from CommonCrawl to training data")
	print("Saved " + str(k - args.cap) + " sentence pairs from CommonCrawl to validation data")
	print("Saved " + str(j - k) + " sentence pairs from CommonCrawl to testing data")

"""
Processes and exports the europarl data in a ready-to-use format
@param args: the argument parser
@param fds: list of files to which the data will be written
@param tokenizer: dict of tokenizers
"""
def processEuroparl(args, fds, tokenizer):
	#Process German data
	print("Processing data from Europarl")
	germanData = []
	englishData = []
	inputFile = open(args.europarlDe, "r")
	inputFileEn = open(args.europarlEn, "r")
	i = 0
	k = args.cap + round(args.cap * 0.01) #Cap for validation set
	j = k + round(args.cap * 0.01) #Cap for testing set
	while 1:
		germanLine = inputFile.readline()
		englishLine = inputFileEn.readline()
		if germanLine == "":
			break #EOF reached
		(germanLine, replacements) = getCleanLineR(germanLine)
		englishLine = getCleanLine(englishLine)
		(germanLine, englishLine) = getFiltered(germanLine, englishLine, tokenizer)
		if germanLine == None or englishLine == None:
			continue
		if i < args.cap:
			fds["trainDe"].write(germanLine + "\n")
			fds["trainEn"].write(englishLine + "\n")
		elif i < k:
			fds["valDe"].write(germanLine + "\n")
			fds["valEn"].write(englishLine + "\n")
		else:
			fds["testDe"].write(germanLine + "\n")
			fds["testEn"].write(englishLine + "\n")
			fds["testR"].write(replacements + "\n")
		i += 1 #Iterate forward
		if i >= j:
			#Finished with the testing set as well
			break
	i = 0
	while 1:
		germanLine = inputFile.readline()
		englishLine = inputFileEn.readline()
		if germanLine == "":
			break #EOF reached
		(germanLine, replacements) = getCleanLineR(germanLine)
		englishLine = getCleanLine(englishLine)
		(germanLine, englishLine) = getFiltered(germanLine, englishLine, tokenizer)
		if germanLine == None or englishLine == None:
			continue
		fds["trainFtDe"].write(germanLine + "\n")
		fds["trainFtEn"].write(englishLine + "\n")
		i += 1 #Iterate forward
		if i >= 16385:
			#Finished with the training set for fine-tuning
			break
	inputFile.close()
	inputFileEn.close()
	print("Saved " + str(args.cap) + " sentence pairs from Europarl to training data")
	print("Saved " + str(k - args.cap) + " sentence pairs from Europarl to validation data")
	print("Saved " + str(j - k) + " sentence pairs from Europarl to testing data")

"""
Processes and saves the data to the respective files
@param args: the argument parser
@param fds: dict of the files to which data will be written
@param tokenizer: dict of the tokenizers
"""
def processParacrawl(args, fds, tokenizer):
	print("Processing data from ParaCrawl")
	germanData = []
	englishData = []
	inputFile = open(args.paracrawl, "r")
	i = 0
	k = args.cap + round(args.cap * 0.01) #Cap for validation set
	j = k + round(args.cap * 0.01) #Cap for testing set
	while 1:
		line = inputFile.readline()
		if line == "":
			#EOF reached
			break
		(germanLine, replacements) = getCleanLineR(line[line.index("\t")+1:])
		englishLine = getCleanLine(line[:line.index("\t")])
		(germanLine, englishLine) = getFiltered(germanLine, englishLine, tokenizer)
		if germanLine == None or englishLine == None:
			continue
		if i < args.cap:
			fds["trainDe"].write(germanLine + "\n")
			fds["trainEn"].write(englishLine + "\n")
		elif i < k:
			fds["valDe"].write(germanLine + "\n")
			fds["valEn"].write(englishLine + "\n")
		else:
			fds["testDe"].write(germanLine + "\n")
			fds["testEn"].write(englishLine + "\n")
			fds["testR"].write(replacements + "\n")
		i += 1 #Iterate forward
		if i >= j:
			#Finished with the testing set as well
			break
	inputFile.close()
	print("Saved " + str(args.cap) + " sentence pairs from ParaCrawl to training data")
	print("Saved " + str(k - args.cap) + " sentence pairs from ParaCrawl to validation data")
	print("Saved " + str(j - k) + " sentence pairs from ParaCrawl to testing data")


def main(args):
	if args.backtranslationsDe == "" or args.backtranslationsEn == "" or args.commoncrawlDe == "" or args.commoncrawlEn == "" or args.europarlDe == "" or args.europarlEn == "" or args.paracrawl == "":
		print("Invalid or no arguments found for dataset paths. Please run with -h flag for usage")
		exit(1)
	print("Preprocess script starting")
	tokenizer = {} #Dict of the tokenizers
	tokenizer["de"] = MosesTokenizer("de")
	tokenizer["en"] = MosesTokenizer("en")
	fds = {} #Holds the files
	fds["trainEn"] = open(TRAIN_DATA_EN, "w+")
	fds["trainDe"] = open(TRAIN_DATA_DE, "w+")
	fds["valEn"] = open(VAL_DATA_EN, "w+")
	fds["valDe"] = open(VAL_DATA_DE, "w+")
	fds["testEn"] = open(TEST_DATA_EN, "w+")
	fds["testDe"] = open(TEST_DATA_DE, "w+")
	fds["testR"] = open(TEST_DATA_REPLACE, "w+")
	fds["trainFtDe"] = open(TRAIN_FT_DE, "w+")
	fds["trainFtEn"] = open(TRAIN_FT_EN, "w+")
	DetectorFactory.seed = 0 #For consistency with LangDetect
	processParacrawl(args, fds, tokenizer)
	processEuroparl(args, fds, tokenizer)
	processCommoncrawl(args, fds, tokenizer)
	processBacktranslations(args, fds, tokenizer)
	#Close the files now that they're finished
	for fd in fds.keys():
		fds[fd].close()
	print("Done with preprocessing script")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Preprocesses data for training a German-to-English machine translation system")
	#parser.add_argument('-t', dest='test', type=bool, default=False, help='This is a test argument, ignore it')
	parser.add_argument('-n', dest='cap', type=int, default=sys.maxsize, help="Limits the amount of samples processed per source for the training data. 1\% of the size will be used for validation and test sets. Default=" + str(sys.maxsize))
	parser.add_argument('--backtranslations_en', dest='backtranslationsEn', type=str, default="", help="Path to the english sentences for backtranslations")
	parser.add_argument('--backtranslations_de', dest='backtranslationsDe', type=str, default="", help="Path to the german sentences for backtranslations")
	parser.add_argument('--europarl_en', dest='europarlEn', type=str, default="", help="Path to the english sentences for Europarl")
	parser.add_argument('--europarl_de', dest='europarlDe', type=str, default="", help="Path to the german sentences for Europarl")
	parser.add_argument('--commoncrawl_en', dest='commoncrawlEn', type=str, default="", help="Path to the english sentences for CommonCrawl")
	parser.add_argument('--commoncrawl_de', dest='commoncrawlDe', type=str, default="", help="Path to the german sentences for CommonCrawl")
	parser.add_argument('--paracrawl', dest='paracrawl', type=str, default="", help="Path to the sentence pairs for ParaCrawl")
	args = parser.parse_args()
	main(args)

"""with open(args.destination + OUTPUT_ID_KEY, "wb") as f:
		pickle.dump(ids, f, pickle.HIGHEST_PROTOCOL)"""