import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Load
df = pd.read_csv('/mnt/Data/Uni/DataEng2/data/raw/Insurance.csv')
df.drop(['id'], axis=1, inplace=True)
df.drop_duplicates(inplace=True)
df = df.reset_index(drop=True)

# Feature identification
cat_features = [i for i in df.columns if df.dtypes[i] == 'object']
ord_feats = ['Vehicle_Age']
nom_feats = cat_features.copy()
nom_feats.remove('Vehicle_Age')

# Encoding
enc = OrdinalEncoder()
df[ord_feats] = enc.fit_transform(df[ord_feats])
for c in nom_feats:
    df[c+'_freq'] = df[c].map(df.groupby(c).size() / df.shape[0])
    indexer = pd.factorize(df[c], sort=True)[1]
    df[c] = indexer.get_indexer(df[c])
df = df.drop(nom_feats, axis=1)

# Save
df.to_csv('/mnt/Data/Uni/DataEng2/data/Lab4/clean_insurance_data.csv', index=False)
print("Saved clean_insurance_data.csv")
