import copy
from typing import *
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util
from torch.utils.data import DataLoader
import torch
import sys
import pickle
from collections import defaultdict
import pandas as pd
from nltk import word_tokenize
sys.path.insert(0, "/home/marcelbraasch/PycharmProjects/argmining-21-keypoint-analysis-sharedtask-code-2/code/src-py")
import sbert_training
work_tokenizer = word_tokenize
device = "cuda:0" if torch.cuda.is_available() else "cpu"
repo_dir = "/home/marcelbraasch/PycharmProjects/argmining-21-keypoint-analysis-sharedtask-code-2/"



def compute_entailment_from_arg_kp(arg, kp, model):
    arg = model.encode(arg, show_progress_bar=False),
    kp = model.encode(kp, show_progress_bar=False)
    return float(util.pytorch_cos_sim(arg, kp))

def create_arg_kps_mapping(arguments_df, key_points_df):
    mapping = {}
    topics = arguments_df["topic"].unique()
    for topic in topics:
        arguments = arguments_df.loc[arguments_df["topic"] == topic][["argument"]].drop_duplicates()
        key_points = key_points_df.loc[key_points_df["topic"] == topic][["key_point"]].drop_duplicates()
        map = pd.merge(arguments, key_points, how="cross")
        mapping[topic] = map
    return mapping


def load_kpm_data(model):

    data = defaultdict(dict)
    for subset in ["dev", "train"]:
        # Load files
        arguments_file = repo_dir + f"data/kpm_data/arguments_{subset}.csv"
        key_points_file = repo_dir + f"data/kpm_data/key_points_{subset}.csv"
        labels_file = repo_dir + f"data/kpm_data/labels_{subset}.csv"
        arguments_df = pd.read_csv(arguments_file)
        key_points_df = pd.read_csv(key_points_file)
        labels_df = pd.read_csv(labels_file)

        # Get gold standard
        positive_labels_df = labels_df.loc[labels_df["label"] == 1]
        gold_standard = pd.merge(positive_labels_df, key_points_df, how="inner", on="key_point_id")
        gold_standard = pd.merge(gold_standard, arguments_df, how="inner", on=["arg_id", "topic"])
        gold_standard = gold_standard[["topic", "argument", "key_point"]]
        gold_standard["score"] = 1
        data[subset]["gold_standard"] = gold_standard

        # Compute model scores
        def compute_score_from(row, model):
            argument = row["argument"]
            key_point = row["key_point"]
            return compute_entailment_from_arg_kp(argument, key_point, model)

        mappings = []
        arg_to_kps = create_arg_kps_mapping(arguments_df, key_points_df)
        for topic, arg_kps_mapping in arg_to_kps.items():
            arg_kps_mapping['score'] = arg_kps_mapping.apply(lambda row: compute_score_from(row, model), axis=1)
            arg_kps_mapping['topic'] = topic
            arg_kps_mapping = arg_kps_mapping[["topic", "argument", "key_point", "score"]]
            mappings.append(arg_kps_mapping)
        predictions = pd.concat(mappings, axis=0)
        data[subset]["predictions"] = predictions

    with open("gold_labels_and_prediction_scores.pkl", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

path = "/home/marcelbraasch/PycharmProjects/argmining-21-keypoint-analysis-sharedtask-code/model"
model = SentenceTransformer(path)
load_kpm_data(model)

"""


[{'dropped': 'Reference', 'new_kp': 'Children can still express themselves using other means'}, {'dropped': 'Children', 'new_kp': 'can still express themselves using other means'}, {'dropped': 'can', 'new_kp': 'Children still express themselves using other means'}, {'dropped': 'still', 'new_kp': 'Children can express themselves using other means'}, {'dropped': 'express', 'new_kp': 'Children can still themselves using other means'}, {'dropped': 'themselves', 'new_kp': 'Children can still express using other means'}, {'dropped': 'using', 'new_kp': 'Children can still express themselves other means'}, {'dropped': 'other', 'new_kp': 'Children can still express themselves using means'}, {'dropped': 'means', 'new_kp': 'Children can still express themselves using other'}]


"""