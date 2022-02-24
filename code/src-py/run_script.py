from sentence_transformers import SentenceTransformer, models, util
from nltk import word_tokenize
from collections import defaultdict
from itertools import chain, combinations
from tqdm import tqdm
import matplotlib.pyplot as plt
import sbert_training
import pandas as pd
import copy
import torch
import sys
import pickle
import random

word_tokenizer = word_tokenize
device = "cuda:0" if torch.cuda.is_available() else "cpu"
repo_dir = "/home/marcelbraasch/PycharmProjects/argmining-21-keypoint-analysis-sharedtask-code-2/"


def save_with_pickle(path, data):
    with open(path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_from_pickle(path):
    data = None
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    return data

def load_model():
    try:
        path = "/home/marcelbraasch/PycharmProjects/argmining-21-keypoint-analysis-sharedtask-code/model"
        model = SentenceTransformer(path)
    except:
        model = sbert_training.train_model('/home/marcelbraasch/PycharmProjects/argmining-21-keypoint-analysis-sharedtask-code-2/data/siamese-data/',
                                           "/home/marcelbraasch/PycharmProjects/argmining-21-keypoint-analysis-sharedtask-code-2/data/kpm_data",
                                           'dev',
                                           "/home/marcelbraasch/PycharmProjects/new_KPA/argmining-21-keypoint-analysis-sharedtask-code-2/code/siamese-models",
                                           'roberta-base',
                                           model_suffix='contrastive-10-epochs',
                                           data_file_suffix='contrastive',
                                          num_epochs=10, max_seq_length=70, add_special_token=True, train_batch_size=32, loss='ContrastiveLoss')
    return model

model = load_model()

def load_closed_class_words():
    path = "/home/marcelbraasch/PycharmProjects/argmining-21-keypoint-analysis-sharedtask-code-2/code/src-py/closed_class_words.txt"
    data = []
    with open(path, "r") as file:
        for line in file:
            data.extend(line.rstrip().split())
    return data

closed_class_words = load_closed_class_words()

def compute_score(arg, kp, model):
    arg = model.encode(arg, show_progress_bar=False),
    kp = model.encode(kp, show_progress_bar=False)
    return float(util.pytorch_cos_sim(arg, kp))

def compute_score_from(row, model):
    argument = row["argument"]
    key_point = row["key_point"]
    return compute_score(argument, key_point, model)

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
    # path = "gold_labels_and_prediction_scores.pkl"
    # try:
    #     return load_from_pickle(path)
    # except:
    #     pass
    data = defaultdict(dict)
    for subset in ["dev"]:#, "train"]:

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
        gold_standard = pd.merge(gold_standard, arguments_df, how="inner", on=["arg_id","topic", "stance"])
        gold_standard = gold_standard.rename(columns={"label": "score"})
        data[subset]["gold_standard"] = gold_standard

        mappings = []
        arg_to_kps = create_arg_kps_mapping(arguments_df, key_points_df)
        for topic, arg_kps_mapping in arg_to_kps.items():
            arg_kps_mapping['score'] = arg_kps_mapping.apply(lambda row: compute_score_from(row), axis=1)
            arg_kps_mapping['topic'] = topic
            arg_kps_mapping = arg_kps_mapping[["topic", "argument", "key_point", "score"]]
            mappings.append(arg_kps_mapping)
        predictions = pd.concat(mappings, axis=0)
        data[subset]["predictions"] = predictions
        save_with_pickle(path, data)
    return data

def tokenize_kp(row):
    return word_tokenizer(row["key_point"])

def _leave_one_out(row):
    words = row["key_point_words"]
    samples = [{"dropped": "Reference", "new_kp": row["key_point"], "score": row["score"]}]
    for i in range(len(words)):
        new_kp = copy.deepcopy(words)
        dropped_word = new_kp.pop(i)
        new_kp = " ".join(new_kp)
        new_score = compute_score(row["argument"], new_kp, model)
        samples.append({"dropped": dropped_word, "new_kp": new_kp, "score": new_score})
    return samples

def leave_one_out(model):
    path = "Data/leave_one_out.pkl"

    try:
        return load_from_pickle(path)
    except:
        pass

    # Iterates over the kpm data dict and compute leave one out for each entry
    dfs = load_kpm_data(model)
    for gold_or_pred in dfs.values():
        for df in gold_or_pred.values():
            df["key_point_words"] = df.apply(lambda row: tokenize_kp(row), axis=1)
            df["leave_one_out"] = df.apply(lambda row: _leave_one_out(row), axis=1)
    save_with_pickle(path, dfs)

    return dfs

def powerset(iterable):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

def _argument_leave_one_out_(tokens, words, drop_size=4):
    samples = []
    lexical_mask = [1 if x not in closed_class_words else 0 for x in tokens]
    lexical_indices = [i for i, x in enumerate(lexical_mask) if x]
    lexical_indices_combinations = powerset(lexical_indices)
    lexical_indices_combinations = [x for x in lexical_indices_combinations
                                    if len(x)<=drop_size][1:]
    for combination in lexical_indices_combinations:
        combination = list(combination)
        combination.sort(reverse=True)
        new_arg = copy.deepcopy(tokens)
        dropped_words = [new_arg.pop(index) for index in combination]
        sample = {"dropped": dropped_words,
                  "new_arg": " ".join(new_arg),
                  "amount_dropped": len(combination),
                  "indices": combination}
        samples.append(sample)
    return samples

def load_mappings(arguments, path="mapping.pkl"):
    try:
        return load_from_pickle(path)
    except:
        mappings = {argument:_argument_leave_one_out_(word_tokenizer(argument), argument)
                    for argument in arguments}
        save_with_pickle(path, mappings)
        return mappings

def compute_dropped_score(row, model):
    argument = row["new_arg"]
    key_point = row["key_point"]
    return compute_score(argument, key_point, model)

"""
1. Analyze per kp their respective argument and check how similar the arguments are
2. Analyze for how many arg-kp pairs the most prevalent words are occuring in both
3. Analyze bad predictions to maybe understand why theyre wrong, or what made the model do the presumably right, but incorrect, prediction
"""

# Create leave one out for the arguments
n = 5
k = 0

# Load data, unique topics and arguments
leave_one_out_path = "Data"
mappings_path = "Data/arg_to_dropped_mapping.pkl"
data_path = "Data/gold_labels_and_prediction_scores.pkl"
data = load_from_pickle(data_path)
predictions = data["train"]["predictions"]
arguments = predictions["argument"].unique()
topics = predictions["topic"].unique()

mappings = load_mappings(arguments[:10])

j = 0

for topic in topics:

    # Get all unique key points for a specific topic
    key_points = predictions.loc[predictions['topic'] == topic]["key_point"].unique()

    for argument in tqdm(arguments, position=0, leave=True):

        # Get the top n key points corresponding to current argument
        top_n = predictions.loc[predictions["argument"]==argument] \
                           .sort_values(by=["score"], ascending=False) \
                           .head(n) \
                           .rename(columns={"score": "reference_score"})

        kps_word_importances = pd.DataFrame()

        for i, key_point in enumerate(top_n.iterrows()):

            key_point = pd.DataFrame(key_point[1]).transpose()

            # Create df from current argument/kp mapping merge with corresponding key_point
            try:
                df = pd.DataFrame.from_dict(mappings[argument])
            except:
                continue
            df = pd.merge(key_point, df, how="cross")

            # Compute dropped scores
            df["dropped_score"] = df.apply(lambda row: compute_dropped_score(row, model), axis=1)
            df["dropped_score_normalized"] = df.apply(lambda row: row["dropped_score"] / row["amount_dropped"], axis=1)
            df["diff"] = df["reference_score"] - df["dropped_score"]
            df["diff_normalized"] = df.apply(lambda row: row["diff"] / row["amount_dropped"], axis=1)

            # Now compute word importance
            words = [x for x in word_tokenizer(df["argument"][0]) if x not in closed_class_words]
            importances = {"word": [], "importance": []}
            for word in words:
                dropped_word_rows = df[df['dropped'].apply(lambda x: word in x)]
                diff_normalized = list(dropped_word_rows["diff_normalized"])
                importances["word"].append(word)
                importances["importance"].append(sum(diff_normalized) / len(diff_normalized))
            importance = pd.DataFrame.from_dict(importances)
            importance["key_point"] = key_point["key_point"].iloc[0]
            kp = importance["key_point"]
            importance.drop(labels=['key_point'], axis=1, inplace=True)
            importance.insert(0, 'key_point', kp)
            importance.rename(columns={"key_point": f"key_point_{i+1}",
                                       "importance": f"importance_{i+1}",
                                       "word": f"word_{i+1}", },
                              inplace=True)
            importance[f"reference_score_{i+1}"] = df["reference_score"][0]
            kps_word_importances = pd.concat((kps_word_importances, importance), axis=1)
            s = 0

        if kps_word_importances is not None:
            kps_word_importances["argument"] = argument
            argument = kps_word_importances['argument']
            kps_word_importances.drop(labels=['argument'], axis=1, inplace=True)
            kps_word_importances.insert(0, 'argument', argument)

            save_with_pickle(f"./LOO_Train/argument_{j}.pkl", kps_word_importances)

            kps_word_importances = None

            j += 1
