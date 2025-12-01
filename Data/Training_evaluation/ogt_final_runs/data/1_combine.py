import pandas as pd
import numpy as np

# read in ogt data
df = pd.read_csv("Temp_per_Genome_final_opt_reg.tsv", sep = "\t", header = 0,  dtype = {'ncbiTaxID_new':str, 'species_id':str, 'superkingdom_id':str,
                     'phylum_id':str, 'class_id':str,'order_id':str, 'family_id':str, 'genus_id':str, 'superkingdom_dummy':str})
orig_cols = df.columns

# read in descriptors per genome
df_desc = pd.read_csv("mean_descriptors.csv", sep = ",", header = 0)

# rename genomes for consistency
df.loc[df.Genome == "GCA_000183325.2", 'Genome'] = "GCF_000183325.1"
df.loc[df.Genome == "GCA_000193435.3", 'Genome'] = "GCF_000193435.2"
df.loc[df.Genome == "GCA_000194075.3", 'Genome'] = "GCF_000194075.2"
df.loc[df.Genome == "GCA_000214035.3", 'Genome'] = "GCF_000214035.2"
df.loc[df.Genome == "GCA_000245095.2", 'Genome'] = "GCF_000245095.1"
df.loc[df.Genome == "GCA_000245115.2", 'Genome'] = "GCF_000245115.1"
df.loc[df.Genome == "GCA_000245135.2", 'Genome'] = "GCF_000245135.1"
df.loc[df.Genome == "GCA_000245155.2", 'Genome'] = "GCF_000245155.1"
df.loc[df.Genome == "GCA_000245175.2", 'Genome'] = "GCF_000245175.1"
df.loc[df.Genome == "GCA_000245195.2", 'Genome'] = "GCF_000245195.1"
df.loc[df.Genome == "GCA_000245215.2", 'Genome'] = "GCF_000245215.1"
df.loc[df.Genome == "GCA_000245235.2", 'Genome'] = "GCF_000245235.1"
df.loc[df.Genome == "GCA_000245255.2", 'Genome'] = "GCF_000245255.1"
df.loc[df.Genome == "GCA_000245275.2", 'Genome'] = "GCF_000245275.1"
df.loc[df.Genome == "GCA_000320505.2", 'Genome'] = "GCF_000320505.1"
df.loc[df.Genome == "GCA_000504565.2", 'Genome'] = "GCF_000504565.1"

# remove the one genome never succesfully fetched
df = df[df.Genome != "GCF_020531845.1"]

# merge:
df = df.merge(df_desc, on = "Genome", how = "left")

# Get all records for which the merge was succesful. Since all calculated descriptors values are not NA, we can filter on any descriptor (here: PP3_mean)
df1 = df[~df["PP3_mean"].isna()]
# Same for unsuccesful cases
df2 = df[df["PP3_mean"].isna()]

# Because unsuccesful merges in our dataset occur due to GCA/GCF synonyms, we can solve this with replacement
df2['Genome'] = df2['Genome'].str.replace("GCA","GCF")

df2 = df2[orig_cols]

df2 = df2.merge(df_desc, on = "Genome", how = "left")

df = pd.concat([df1,df2], ignore_index = True).reset_index(drop = True)

# Write to output
df.to_csv("Input_opt.tsv", sep = "\t", index = False)

