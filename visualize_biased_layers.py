import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("gender_bbq/results/attention_results/attention_per_layer_COT.csv")

df["attn_diff_list"] = df["avg_attn_difference"].apply(lambda x: list(map(float, x.split(","))))

layer_df = pd.DataFrame(df["attn_diff_list"].to_list())
layer_df.index = df["prompt_index"]  # Set prompt index as row index
# Normalize each row (prompt) by its max difference across all layers
normalized_df = layer_df.div(layer_df.max(axis=1), axis=0)
normalized_df = normalized_df.fillna(0)


# Optional: Rename columns to indicate layer numbers
layer_df.columns = [f"layer_{i}" for i in range(layer_df.shape[1])]

'''
plt.figure(figsize=(14, 6))

for i in range(layer_df.shape[1]):
    sns.kdeplot(layer_df.iloc[:, i], label=f'Layer {i}', alpha=0.3)

plt.xlabel("Attention Difference")
plt.ylabel("Density")
plt.title("Layer-wise Attention Difference Distribution (No CoT)")
plt.legend(loc="upper right", ncol=2)
plt.tight_layout()
plt.savefig("attention_diff_layerwise_kde_no_COT.png", dpi=300)
plt.show()
'''

# Sum attention differences across layers for each prompt
total_diff_per_prompt = layer_df.sum(axis=1)


# Plot histogram
plt.figure(figsize=(10, 6))
sns.histplot(total_diff_per_prompt, bins=30, kde=True, color="steelblue")
plt.ylim(0, 1000)
plt.xlabel("Sum of Attention Differences (Stereotypical – Anti-Stereotypical)")
plt.ylabel("Number of Prompts")
plt.title("Distribution of Total Attention Differences per Prompt (CoT)")
#plt.xlim(right=1000)  # Set max x-axis value to 1000
plt.tight_layout()
plt.savefig("attention_diff_histogram_COT.png", dpi=300)
plt.show()



# Plot the heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(normalized_df.T, cmap="Greens", yticklabels=True)

plt.xlabel("Prompt Index")
plt.ylabel("Layer")
plt.title("Absolute Attention Difference per Layer and Prompt with CoT")
plt.tight_layout()

# Save the heatmap to file
plt.savefig("attention_difference_heatmap_COT.png", dpi=300)  # Change filename/format as needed
plt.show()
