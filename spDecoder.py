#!/usr/bin/python

import argparse #Arguments and flags
import sentencepiece as spm #Subword segmentation

def convertToInt(strList):
	intList = []
	for string in strList:
		if string == "<unk>":
			intList.append(0)
			continue
		intList.append(int(string))
	return intList

def main(args):
	if args.input == "" or args.output == "" or args.model == "":
		print("Incorrect usage. Please see spDecoder.py -h")
		exit(1)
	model = spm.SentencePieceProcessor(model_file=args.model)
	inputFile = open(args.input, "r")
	outputFile = open(args.output, "w+")
	while 1:
		line = inputFile.readline()
		if line == "":
			break #EOF reached
		sample = line.split()
		sample = convertToInt(sample)
		line = model.decode(sample)
		outputFile.write(line)
		outputFile.write("\n")
	inputFile.close()
	outputFile.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Python script wrapper for spm_decode command in the terminal")
	#parser.add_argument('-t', dest='test', type=bool, default=False, help='This is a test argument, ignore it')
	parser.add_argument('--input', dest='input', type=str, default="", help="Data for decoding")
	parser.add_argument('--output', dest='output', type=str, default="", help="Output for decoded data")
	parser.add_argument('--model', dest='model', type=str, default="", help="The trained model to use")
	args = parser.parse_args()
	main(args)