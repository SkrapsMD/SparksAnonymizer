from pydataset import data
import statsmodels.api as sm

# Import Sparks Anonymizer and AnonymizeConfig
from sparks_anonymizer import SparksAnonymizer, AnonymizeConfig, rename_to_anonymized, rename_to_original

df = data("Boston")
print(df.head())

config = AnonymizeConfig(
    seed=42,
    string_strategy="stable_pseudonym")

anonymizer = SparksAnonymizer(config)
df, name_map = anonymizer.anonymize(df)
print(df.head())

# Now rename and run LLM Generated code
""" I entered this prompt: 
I have this data: "    col_float64_1  col_float64_2  col_float64_3  col_int64_1  col_float64_4  col_float64_5  col_float64_6  col_float64_7  col_int64_2  col_int64_3  col_float64_8  col_float64_9  col_float64_10  col_float64_11
1      68.865097      31.445069      13.357547            1       0.761367       4.689710      37.021994       5.895370            3          363      19.310290     236.167186       24.858612       23.452273
2      39.053282      83.738236      26.377667            0       0.691623       8.051145      23.935107       5.008700           16          379      21.974721     173.326645       14.233814       36.317012
3      76.395674      49.414165       9.809228            0       0.813737       6.944988      70.082043       9.052417           14          230      15.892800     229.946209       16.083176       43.048397
4      62.051070      11.585672       5.294984            0       0.695295       5.754141      69.593093       7.497098           16          563      12.714825     205.552230        3.180733       37.441319
5       8.385267       7.205915      18.958406            0       0.638753       4.542716      55.253241       6.535300           12          418      13.780875     270.377734       34.996236       12.829440" 
Please run a regression using this data. 

"""
# Rename to anonymized names so we can run the LLM generated code 
rename_to_anonymized(df, name_map)

# dependent variable
y = df["col_float64_11"]

# regressors
X = df.drop(columns=["col_float64_11", "col_int64_3"])
X = sm.add_constant(X)

# weights
weights = df["col_int64_3"]

# WLS regression
model = sm.WLS(y, X, weights=weights)
res = model.fit()

print(res.summary())

# Revert back to the original naming scheme
rename_to_original(df, name_map)