import numpy as np
import pandas as pd
import apyori
from apyori import apriori
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_excel('/apriori.xlsx')
frq_items = apriori(data, min_support = 0.6, use_colnames = True)
rules = association_rules(frq_items, metric = "confidence", min_threshold = 0.80) 
print(rules)
