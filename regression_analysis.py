"""Regression analysis to evaluate the dependence of pool- and test-user tracking duration on the
matching performance (see table 1 in paper)"""

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import statsmodels.api as sm

study = "gc1"
out_df = {}
p_value_dict = {}
row_dict_list = []

for k in [0, 5, 10]:
    res = pd.read_csv(f"./outputs/acc_k{k}/mean_mse_combined.csv", index_col="p_duration")
    res_arr = np.array(res)
    months_p, months_u = np.where(res_arr)
    diff = np.abs(months_p - months_u)
    X = (np.array(list(zip(months_p + 1, months_u+1, diff)))) * 4
    Y = res_arr.flatten()
    X = X[~np.isnan(Y)]
    Y = Y[~np.isnan(Y)]
    reg = LinearRegression().fit(X, Y)

    temp_list = reg.coef_.tolist() + [reg.intercept_]

    # todo: results of table 1 are calculated but not stored
    row_dict_list.append("Coefficient pool duration")



    X2 = sm.add_constant(X)
    est = sm.OLS(Y, X2)
    est2 = est.fit()

    p_values = est2.pvalues.tolist()
    p_value_dict[k] = p_values



df = pd.DataFrame(out_df).rename(columns={0: "MRR", 1: "1-Accuracy", 5: "5-Accuracy", 10:"10-Accuracy"},
                            index={0:"Coefficient pool duration", 1: "Coefficient test duration", 2: "Coefficient of absolute difference between pool and test duration", 3: "Intercept"}).swapaxes(1,0)


