#!/usr/bin/python

import argparse #Arguments and flags
import sentencepiece as spm #Subword segmentation



def main(args):
	if args.input == "" or args.modelPrefix == "" or args.vocab_size == 0 or args.charCoverage == 0:
		print("Incorrect usage. Please see spTrainer.py -h")
		exit(1)
	if args.userSymb != "":
		spm.SentencePieceTrainer.train(input=args.input, model_prefix=args.model_prefix, vocab_size=args.vocab_size, character_coverage=args.charCoverage, user_defined_symbols=args.userSymb)
	else:
		spm.SentencePieceTrainer.train(input=args.input, model_prefix=args.model_prefix, vocab_size=args.vocab_size, character_coverage=args.charCoverage)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Python script wrapper for spm_train command in the terminal")
	#parser.add_argument('-t', dest='test', type=bool, default=False, help='This is a test argument, ignore it')
	parser.add_argument('--input', dest='input', type=str, default="", help="Data for training the model")
	parser.add_argument('--model_prefix', dest='modelPrefix', type=str, default="", help="Model prefix")
	parser.add_argument('--vocab_size', dest='vocabSize', type=int, default=0, help="Vocabulary size")
	parser.add_argument('--character_coverage', dest='charCoverage', type=int, default=0, help="Character coverage")
	parser.add_argument('--user_defined_symbols', dest='userSymb', type=str, default="", help="User defined symbols separated by commas")
	args = parser.parse_args()
	main(args)

"""with open(args.destination + OUTPUT_ID_KEY, "wb") as f:
		pickle.dump(ids, f, pickle.HIGHEST_PROTOCOL)"""