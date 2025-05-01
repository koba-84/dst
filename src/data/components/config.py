import os
import logging 
import argparse
from tqdm import tqdm
import torch

PAD_token = 1
SOS_token = 3
EOS_token = 2
UNK_token = 0 

if torch.cuda.is_available():
    USE_CUDA = True
else:
    USE_CUDA = False

MAX_LENGTH = 10
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
parser = argparse.ArgumentParser(description='Multi-Domain DST')


#args = vars(parser.parse_args())
"""
# Training Setting
parser.add_argument('-t','--task', help='Task Number', required=False, default="dst")
parser.add_argument('--train', help='Set to "yes" to train, "no" to skip training', type=str, choices=["yes", "no"], default="yes")
parser.add_argument('--train_batch_size', help='Train Batch_size', required=True, type=int)
parser.add_argument('--save_dir', help='dir of the file to save', required=True, type=str)
parser.add_argument('-all_vocab','--all_vocab', help='', required=False, default=1, type=int)
parser.add_argument('-data_ratio','--data_ratio', help='', required=False, default=100, type=int)
parser.add_argument('-imbsamp','--imbalance_sampler', help='', required=False, default=0, type=int)
parser.add_argument('--num_workers', help='number of workers', required=False, default=4, type=int)
parser.add_argument('-sample','--sample', help='Number of Samples', required=False,default=None)
parser.add_argument('-patience','--patience', help='', required=False, default=6, type=int)
parser.add_argument('-es','--earlyStop', help='Early Stop Criteria, BLEU or ENTF1', required=False, default='BLEU')
parser.add_argument('-um','--unk_mask', help='mask out input token to UNK', type=int, required=False, default=1)

# Testing Setting
parser.add_argument('-rundev','--run_dev_testing', help='', required=False, default=0, type=int)
parser.add_argument('-viz','--vizualization', help='vizualization', type=int, required=False, default=0)
parser.add_argument('-gs','--genSample', help='Generate Sample', type=int, required=False, default=0)
parser.add_argument('-evalp','--evalp', help='evaluation period', required=False, default=1)
parser.add_argument('-an','--addName', help='An add name for the save folder', required=False, default='')
parser.add_argument('-eb','--eval_batch', help='Evaluation Batch_size', required=False, type=int, default=0)
parser.add_argument('--eval_batch_size', help='Eval Batch_size', required=True, type=int)
# Model architecture
parser.add_argument('-gate','--use_gate', help='', required=False, default=1, type=bool)
parser.add_argument('-le','--load_embedding', help='', required=False, default=0, type=int)
parser.add_argument('-femb','--fix_embedding', help='', required=False, default=0, type=int)
parser.add_argument('-paral','--parallel_decode', help='', required=False, default=0, type=int)

# Model Hyper-Parameters
parser.add_argument('-dec','--decoder', help='decoder model', required=False)
parser.add_argument('-hdd','--hidden', help='Hidden size', required=False, type=int, default=400)
parser.add_argument('-lr','--learn', help='Learning Rate', required=False, type=float)
parser.add_argument('-dr','--drop', help='Drop Out', required=False, type=float)
parser.add_argument('-lm','--limit', help='Word Limit', required=False,default=-10000)
parser.add_argument('-clip','--clip', help='gradient clipping', required=False, default=10, type=int) 
parser.add_argument('-tfr','--teacher_forcing_ratio', help='teacher_forcing_ratio', type=float, required=False, default=0.5)
parser.add_argument('-l','--layer', help='Layer Number', required=False)
Model Hyper-Parameters
parser.add_argument('--dataset', help='dataset', required=True, type=str)
parser.add_argument('--decoder', help='decoder model', required=False)
# Unseen Domain Setting
parser.add_argument('-exceptd','--except_domain', help='', required=False, default="", type=str)
parser.add_argument('-onlyd','--only_domain', help='', required=False, default="", type=str)
parser.add_argument('-fisher_sample','--fisher_sample', help='number of sample used to approximate fisher mat', type=int, required=False, default=0)
parser.add_argument('-l_ewc','--lambda_ewc', help='regularization term for EWC loss', type=float, required=False, default=0.01)
parser.add_argument("--all_model", action="store_true")
parser.add_argument("--domain_as_task", action="store_true")
parser.add_argument('--run_except_4d', help='', required=False, default=1, type=int)
parser.add_argument("--strict_domain", action="store_true")




args = vars(parser.parse_args())
if args["load_embedding"]:
    args["hidden"] = 400
    print("[Warning] Using hidden size = 400 for pretrained word embedding (300 + 100)...")
if args["fix_embedding"]:
    args["addName"] += "FixEmb"
if args["except_domain"] != "":
    args["addName"] += "Except"+args["except_domain"]
if args["only_domain"] != "":
    args["addName"] += "Only"+args["only_domain"]

print(str(args))
"""


def fix_general_label_error(labels, type, slots):
    label_dict = dict([ (l[0], l[1]) for l in labels]) if type else dict([ (l["slots"][0][0], l["slots"][0][1]) for l in labels]) 

    GENERAL_TYPO = {
        # type
        "guesthouse":"guest house", "guesthouses":"guest house", "guest":"guest house", "mutiple sports":"multiple sports", 
        "sports":"multiple sports", "mutliple sports":"multiple sports","swimmingpool":"swimming pool", "concerthall":"concert hall", 
        "concert":"concert hall", "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture", 
        "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", "churches":"church",
        # area
        "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north", "cen":"centre", "east side":"east", 
        "east area":"east", "west part of town":"west", "ce":"centre",  "town center":"centre", "centre of cambridge":"centre", 
        "city center":"centre", "the south":"south", "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north", 
        "centre of town":"centre", "cb30aq": "none",
        # price
        "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate", 
        # day
        "next friday":"friday", "monda": "monday", 
        # parking
        "free parking":"free",
        # internet
        "free internet":"yes",
        # star
        "4 star":"4", "4 stars":"4", "0 star rarting":"none",
        # others 
        "y":"yes", "any":"dontcare", "n":"no", "does not care":"dontcare", "not men":"none", "not":"none", "not mentioned":"none",
        '':"none", "not mendtioned":"none", "3 .":"3", "does not":"no", "fun":"none", "art":"none",  
        }

    for slot in slots:
        if slot in label_dict.keys():
            # general typos
            if label_dict[slot] in GENERAL_TYPO.keys():
                label_dict[slot] = label_dict[slot].replace(label_dict[slot], GENERAL_TYPO[label_dict[slot]])
            
            # miss match slot and value 
            if  slot == "hotel-type" and label_dict[slot] in ["nigh", "moderate -ly priced", "bed and breakfast", "centre", "venetian", "intern", "a cheap -er hotel"] or \
                slot == "hotel-internet" and label_dict[slot] == "4" or \
                slot == "hotel-pricerange" and label_dict[slot] == "2" or \
                slot == "attraction-type" and label_dict[slot] in ["gastropub", "la raza", "galleria", "gallery", "science", "m"] or \
                "area" in slot and label_dict[slot] in ["moderate"] or \
                "day" in slot and label_dict[slot] == "t":
                label_dict[slot] = "none"
            elif slot == "hotel-type" and label_dict[slot] in ["hotel with free parking and free wifi", "4", "3 star hotel"]:
                label_dict[slot] = "hotel"
            elif slot == "hotel-star" and label_dict[slot] == "3 star hotel":
                label_dict[slot] = "3"
            elif "area" in slot:
                if label_dict[slot] == "no": label_dict[slot] = "north"
                elif label_dict[slot] == "we": label_dict[slot] = "west"
                elif label_dict[slot] == "cent": label_dict[slot] = "centre"
            elif "day" in slot:
                if label_dict[slot] == "we": label_dict[slot] = "wednesday"
                elif label_dict[slot] == "no": label_dict[slot] = "none"
            elif "price" in slot and label_dict[slot] == "ch":
                label_dict[slot] = "cheap"
            elif "internet" in slot and label_dict[slot] == "free":
                label_dict[slot] = "yes"

            # some out-of-define classification slot values
            if  slot == "restaurant-area" and label_dict[slot] in ["stansted airport", "cambridge", "silver street"] or \
                slot == "attraction-area" and label_dict[slot] in ["norwich", "ely", "museum", "same area as hotel"]:
                label_dict[slot] = "none"

    return label_dict


