# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:29:02 2020

@author: lenovo
"""

import pickle
import numpy as np
import pandas as pd

rs = pickle.load(open("./ensemble_model_ncVSad.pickle.dat", "rb"))

print(rs.keys())


mods = rs["merged_model"]

em = mods.estimators_

coef = []
for pipe in em:
    print(pipe[0])
    clf = pipe["classify"] 
    if hasattr(clf, "coef_"): 
        coef_ =  clf.coef_
        coef_ = pipe[0].inverse_transform(coef_)
        coef.append(coef_)
    # else:
    #     coef.append(np.zeros([1,10200]))

mean_coef = np.mean(coef,axis=0)[0]
mean_coef=np.sort(mean_coef)
mean_coef = mean_coef[::-1]
np.savetxt("coef.txt", mean_coef)

# fm = mods.final_estimator_
# stack_coef = fm.coef_

coef = pd.read_csv("coef.txt", header=None)
fn = pd.read_csv("fn.txt", header=None)
sc = pd.concat([fn, coef], axis=1)
sc.columns = ["feature_name","weight"]
sc.to_excel("coef.xlsx", index=False)
