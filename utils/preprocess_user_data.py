import pandas as pd
from utils.preprocess_text import preprocess
import os
TF_ENABLE_ONEDNN_OPTS='0'
import tensorflow
import pandas as pd
import json
import csv
import os
from tensorflow.data import Dataset
from utils.tokenizer import call_tokenizer
def auto_detect_filter_data(input_path, output_path):
    # Đọc dữ liệu từ file vào một DataFrame
    df = pd.read_csv(input_path)
    
    # Giả định cột đánh giá là cột văn bản có độ dài trung bình cao nhất
    text_length = df.applymap(lambda x: len(str(x))).mean()
    review_column = text_length.idxmax()
    # Lọc và lưu cột đánh giá
    filtered_df = df[[review_column]]
    filtered_df = filtered_df.dropna()
    filtered_df.to_csv(output_path, index=False)
def preprocess_data(df):
    for column in df.columns:
        df[column] = df[column].apply(preprocess)
    return df
def keep_longest_average_columns(input_file, output_file):
    # Xác định loại file đầu vào (CSV, JSON, XLSX hoặc XLSM) dựa vào phần mở rộng của tên file
    input_extension = os.path.splitext(input_file)[1].lower()

    if input_extension == '.csv':
        with open(input_file, 'r', newline='', encoding='utf-8') as f_in, open(output_file, 'w', newline='', encoding='utf-8') as f_out:
            csv_reader = csv.reader(f_in)
            csv_writer = csv.writer(f_out)

            for row in csv_reader:
                row_length = len(row)
                column_lengths = [len(cell) for cell in row]
                average_length = sum(column_lengths) / row_length
                longest_average_length = max(average_length for average_length in column_lengths)
                longest_average_cells = [cell for cell in row if len(cell) == longest_average_length]
                csv_writer.writerow(longest_average_cells)
    elif input_extension in ['.xlsx', '.xlsm']:
        df = pd.read_excel(input_file, engine='openpyxl', encoding='ISO-8859-1')
        max_average_length = df.apply(lambda row: sum(len(str(cell)) for cell in row) / len(row), axis=1).max()
        
        with open(output_file, 'w', newline='', encoding='ISO-8859-1') as f_csv:
            csv_writer = csv.writer(f_csv)
            
            for _, row in df.iterrows():
                longest_average_values = [cell for cell in row if len(str(cell)) == max_average_length]
                csv_writer.writerow(longest_average_values)
                
                
    elif input_extension == '.json':
        with open(input_file, 'r', encoding='utf-8') as f_json, open(output_file, 'w', newline='', encoding='utf-8') as f_csv:
            csv_writer = csv.writer(f_csv)
            for line in f_json:
                data = json.loads(line)
                row_length = len(data)
                column_lengths = [len(str(value)) for value in data.values()]
                average_length = sum(column_lengths) / row_length
                longest_average_length = max(average_length for average_length in column_lengths)
                longest_average_values = [value for value in data.values() if len(str(value)) == longest_average_length]
                csv_writer.writerow(longest_average_values)