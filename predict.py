from tensorflow.train import latest_checkpoint
from tensorflow.keras.models import load_model
from create_model import create_model
from utils.config import MODEL_PATH
from transformers import TFAutoModel
from utils.tokenizer import PRETRAINED_MODEL
from utils.variables import df_test, tokenizer
from utils.preprocess_text import preprocess
import numpy as np
import pandas as pd
from tensorflow.data import Dataset

#todo
from utils.preprocess_user_data import preprocess_data

pretrained_bert = TFAutoModel.from_pretrained(PRETRAINED_MODEL, output_hidden_states=True)
reloaded_model = create_model(pretrained_bert)
# reloaded_model.load_weights("F:/jvb/Sentiment-Analysis-leanhtu2/Sentiment-Analysis-leanhtu/model/4_12_2.h5")
reloaded_model.load_weights(f"{MODEL_PATH}/4_12_2.h5")
replacements = {0: None, 3: 'positive', 1: 'negative', 2: 'neutral'}
categories = df_test.columns[1:]


def print_acsa_pred(replacements, categories, sentence_pred, confidence_scores):
    sentiments = map(lambda x: replacements[x], sentence_pred)
    results = []
    for category, sentiment, confidence in zip(categories, sentiments, confidence_scores):
        if sentiment: 
            results.append(f'{category},{sentiment},{confidence:.2f}')
    return results if results else None  

def predict_text(model, inputs, batch_size=1, verbose=0):
    y_pred = model.predict(inputs, batch_size=batch_size, verbose=verbose)
    y_pred = y_pred.reshape(len(y_pred), -1, 4)
    return np.argmax(y_pred, axis=-1), np.max(y_pred, axis=-1)  

def show_predict_text(text):
    text = preprocess(text)
    tokenized_input = tokenizer(text, padding='max_length', truncation=True)
    features = {x: [[tokenized_input[x]]] for x in tokenizer.model_input_names}
    pred, confidences = predict_text(reloaded_model, Dataset.from_tensor_slices(features))
    results = []
    for i in range(len(pred)):
        absa_pred = print_acsa_pred(replacements, categories, pred[i], confidences[i])
        if(absa_pred != None):
            for i in range(len(absa_pred)):
                parts = absa_pred[i].split(',')
                positive_value = parts[1]
                confidences_value = float(parts[2])
                if (positive_value == 'positive' and confidences_value >= 0.6):
                    parts.append('⭐️⭐️⭐️⭐️⭐️')
                if (positive_value == 'positive' and  confidences_value < 0.6):
                    parts.append('⭐️⭐️⭐️⭐️')
                elif positive_value == 'neutral':
                    parts.append('⭐️⭐️⭐️')
                elif (positive_value == 'negative'and confidences_value < 0.6):
                    parts.append('⭐️⭐️')
                elif (positive_value == 'negative'and confidences_value >= 0.6):
                    parts.append('⭐️')
                absa_pred[i] = ','.join(parts)
            results.append(absa_pred)
        else:
            results.append(absa_pred)
    return results
    


def predict_csv(model, df):
    input_sentences = df.iloc[:, 0].tolist()
    tokenized_inputs = tokenizer(input_sentences, padding='max_length', truncation=True)
    features = {x: [tokenized_inputs[x]] for x in tokenizer.model_input_names}
    pred, confidences = predict_text(model, Dataset.from_tensor_slices(features))
    return pred, confidences

def show_predict_csv(df_clean, output_csv_path):
    pred, confidences = predict_csv(reloaded_model, df_clean)
    results = []
    for i in range(len(pred)):
        absa_pred = print_acsa_pred(replacements, categories, pred[i], confidences[i])
        if(absa_pred != None):
            for i in range(len(absa_pred)):
                parts = absa_pred[i].split(',')
                positive_value = parts[1]
                confidences_value = float(parts[2])
                if (positive_value == 'positive' and confidences_value >= 0.6):
                    parts.append('⭐️⭐️⭐️⭐️⭐️')
                if (positive_value == 'positive' and  confidences_value < 0.6):
                    parts.append('⭐️⭐️⭐️⭐️')
                elif positive_value == 'neutral':
                    parts.append('⭐️⭐️⭐️')
                elif (positive_value == 'negative'and confidences_value < 0.6):
                    parts.append('⭐️⭐️')
                elif (positive_value == 'negative'and confidences_value >= 0.6):
                    parts.append('⭐️')
                absa_pred[i] = ','.join(parts)
            results.append(absa_pred)
        else:
            results.append(absa_pred)

    df_source = pd.read_csv("data_user/raw.csv")
    df_input_with_pred = df_source.copy()
    df_input_with_pred['label'] = results
    df_input_with_pred.to_csv(output_csv_path, index=False)


# datavao = "data_user/test.csv"
# df_clean = pd.read_csv(datavao, index_col=None)
# datara = "data_user/chuachuan.csv"
# show_predict_csv(df_clean,datara)