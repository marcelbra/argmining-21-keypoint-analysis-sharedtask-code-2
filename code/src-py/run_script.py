from sentence_transformers import SentenceTransformer, models, util
from nltk import word_tokenize
from collections import defaultdict
import sbert_training
import pandas as pd
import copy
import torch
import sys
import pickle

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

    path = "gold_labels_and_prediction_scores.pkl"

    try:
        return load_from_pickle(path)
    except:
        pass

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
        gold_standard = pd.merge(gold_standard, arguments_df, how="inner", on=["arg_id","topic"])
        gold_standard = gold_standard[["topic", "argument", "key_point"]]
        gold_standard["score"] = 1
        data[subset]["gold_standard"] = gold_standard

        # Compute model scores
        def compute_score_from(row):
            argument = row["argument"]
            key_point = row["key_point"]
            return compute_entailment(argument, key_point, model)

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

def compute_entailment(arg, kp, model):
    arg = model.encode(arg, show_progress_bar=False),
    kp = model.encode(kp, show_progress_bar=False)
    return float(util.pytorch_cos_sim(arg, kp))

def tokenize_kp(row):
    return word_tokenizer(row["key_point"])

def _leave_one_out(row):
    words = row["key_point_words"]
    samples = [{"dropped": "Reference", "new_kp": row["key_point"], "score": row["score"]}]
    for i in range(len(words)):
        new_kp = copy.deepcopy(words)
        dropped_word = new_kp.pop(i)
        new_kp = " ".join(new_kp)
        new_score = compute_entailment(row["argument"], new_kp, model)
        samples.append({"dropped": dropped_word, "new_kp": new_kp, "score": new_score})
    return samples

def leave_one_out(model):
    path = "leave_one_out.pkl"

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

model = load_model()
leave_one_out(model)

