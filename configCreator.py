#!/usr/bin/python

import argparse #Arguments and flags
import sentencepiece as spm #Subword segmentation



def main(args):
	if args.tmpt == "" or args.output == "":
		print("Incorrect usage. Template and output arguments are required. Please see spEncoder.py -h")
		exit(1)
	inputFile = open(args.tmpt, "r")
	template = inputFile.read()
	inputFile.close()
	if args.src_vocab != "":
		template = template.replace("system_data.vocab.src", args.src_vocab)
	if args.tgt_vocab != "":
		template = template.replace("system_data.vocab.tgt", args.tgt_vocab)
	if args.save_data != "":
		template = template.replace("save_data: system_data", "save_data: " + args.save_data)
	if args.train_path_src != "":
		template = template.replace("path_src: trainDataDe.txt", "path_src: " + args.train_path_src)
	if args.train_path_tgt != "":
		template = template.replace("path_tgt: trainDataEn.txt", "path_tgt: " + args.train_path_tgt)
	if args.val_path_src != "":
		template = template.replace("path_src: valDataDe.txt", "path_src: " + args.val_path_src)
	if args.val_path_tgt != "":
		template = template.replace("path_tgt: valDataEn.txt", "path_tgt: " + args.val_path_tgt)
	if args.save_model != "":
		template = template.replace("foo", args.save_model)
	if args.n_layers != "":
		template = template.replace("enc_layers: 6", "enc_layers: " + args.n_layers)
		template = template.replace("dec_layers: 6", "dec_layers: " + args.n_layers)
	if args.lr != "":
		template = template.replace("learning_rate: 2", "learning_rate: " + args.lr)
	if args.ff != "":
		template = template.replace("transformer_ff: 2048", "transformer_ff: " + args.ff)
	if args.ini != "":
		template = template.replace("param_init: 0", "param_init: " + args.ini)
	outputFile = open(args.output, "w+")
	outputFile.write(template)
	outputFile.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Makes a config file for the OpenNMT-py model")
	#parser.add_argument('-t', dest='test', type=bool, default=False, help='This is a test argument, ignore it')
	parser.add_argument('-t', dest='tmpt', type=str, default="", help="Template for the config file")
	parser.add_argument('-o', dest='output', type=str, default="", help="Output for config file")
	parser.add_argument('--src_vocab', dest='src_vocab', type=str, default="", help="The vocab for the source language")
	parser.add_argument('--tgt_vocab', dest='tgt_vocab', type=str, default="", help="The vocab for the target side")
	parser.add_argument('--save_data', dest='save_data', type=str, default="", help="Where save_data is written when building vocab")
	parser.add_argument('--train_path_src', dest='train_path_src', type=str, default="", help="The training data for source side")
	parser.add_argument('--train_path_tgt', dest='train_path_tgt', type=str, default="", help="The training data for target side")
	parser.add_argument('--val_path_src', dest='val_path_src', type=str, default="", help="The validaiton data for source side")
	parser.add_argument('--val_path_tgt', dest='val_path_tgt', type=str, default="", help="The validation data for target side")
	parser.add_argument('--save_model', dest='save_model', type=str, default="", help="Where the trained model is saved")
	parser.add_argument('--n_layers', dest='n_layers', type=str, default="", help="Number of encoder and decoder layers each")
	parser.add_argument('--lr', dest='lr', type=str, default="", help="Learning rate")
	parser.add_argument('--ff', dest='ff', type=str, default="", help="Number of hidden transformer feed-forward layers")
	parser.add_argument('--ini', dest='ini', type=str, default="", help="Seed for initialization of weights")
	args = parser.parse_args()
	main(args)