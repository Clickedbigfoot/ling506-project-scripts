#!/usr/bin/python

import argparse #Arguments and flags
import sentencepiece as spm #Subword segmentation



def main(args):
	if args.input == "" or args.output == "" or args.model == "":
		print("Incorrect usage. Please see spEncoder.py -h")
		exit(1)
	model = spm.SentencePieceProcessor(model_file=args.model)
	inputFile = open(args.input, "r")
	outputFile = open(args.output, "w+")
	while 1:
		line = inputFile.readline().replace("\n", "")
		if line == "":
			break #EOF reached
		line = model.encode(line)
		for token in line:
			outputFile.write(str(token) + " ")
		outputFile.write("\n")
	inputFile.close()
	outputFile.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Python script wrapper for spm_train command in the terminal")
	#parser.add_argument('-t', dest='test', type=bool, default=False, help='This is a test argument, ignore it')
	parser.add_argument('--input', dest='input', type=str, default="", help="Data for encoding")
	parser.add_argument('--output', dest='output', type=str, default="", help="Output for encoded data")
	parser.add_argument('--model', dest='model', type=str, default="", help="The trained model to use")
	args = parser.parse_args()
	main(args)