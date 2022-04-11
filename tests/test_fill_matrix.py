from fill_matrix import calculate_reciprocal_rank, calculate_topk_accuracy

import pandas as pd
import os

k = 2

df = pd.read_csv(os.path.join("..", "test_data_matrix_aggregation.csv"), sep=";")
df['same_user'] = df['u_user_id'] == df['p_user_id']

mean_matrix, std_matrix = calculate_topk_accuracy(df, k)
print(mean_matrix)
print(std_matrix)
mean_matrix, std_matrix = calculate_reciprocal_rank(df, k=k, distance_column='distance')

print(mean_matrix)
print(std_matrix)