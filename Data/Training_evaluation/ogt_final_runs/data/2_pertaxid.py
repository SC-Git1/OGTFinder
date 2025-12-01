import pandas as pd


df = pd.read_csv("Input_opt.tsv", sep = "\t", dtype = {'ncbiTaxID_new':str, 'species_id':str, 'superkingdom_id':str,'phylum_id':str, 'class_id':str,
                                                          'order_id':str, 'family_id':str, 'genus_id':str, 'superkingdom_dummy':str})

print(len(df["ncbiTaxID_new"].unique()))

# groupby, then average

meancols = [col for col in df.columns if col.endswith("_mean")]

dictmean = {}
for col in meancols:
    dictmean[col] = 'mean'

dictmeta = {'ncbiTaxID_new':'first', 'species_id':'first','species':'first', 'superkingdom_id':'first', 'superkingdom':'first',
       'phylum_id':'first', 'phylum':'first', 'class_id':'first', 'class':'first', 'order_id':'first', 'order':'first',
       'family_id':'first', 'family':'first', 'genus_id':'first', 'genus':'first','median_temp':'first', 'superkingdom_dummy':'first'}


df = df.groupby('ncbiTaxID_new').agg(dictmeta | dictmean)

print(len(df))
print(df.head())

df.to_csv("Input_opt_uniqtaxids.tsv", sep = "\t", index = False)
