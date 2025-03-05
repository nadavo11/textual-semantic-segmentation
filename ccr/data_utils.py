import pandas as pd

def rename_qid_cols(data_df):
    # Load the questionnaire mapping
    questionnaire_df = pd.read_csv("values_questionnaire.csv")

    # Extract question IDs (assuming they are ordered correctly)
    question_ids = questionnaire_df["question_id"].tolist()

    # Generate the original column names (assuming they are like 'sim_item_1', 'sim_item_2', ...)
    original_columns = [col for col in data_df.columns if col.startswith("sim_item_")]

    # Create a mapping: 'sim_item_1' -> first question_id, 'sim_item_2' -> second question_id, etc.
    rename_mapping = {orig_col: f"sim_item_{str(qid)}" for orig_col, qid in zip(original_columns, question_ids)}

    # Rename columns
    data_df.rename(columns=rename_mapping, inplace=True)

    # Save the updated file
    data_df.to_csv("updated_table.csv", index=False)

    print("✅ Column names updated successfully! Saved as 'updated_table.csv'.")

def compute_dimension_scores(ccr_df, dimension_map):
    from collections import defaultdict
    dim2ids = defaultdict(list)
    print(dimension_map)

    # Invert: dimension -> list of question IDs
    for q_id, dim_name in dimension_map.items():
        dim2ids[dim_name].append(q_id)

    # For each dimension, average the relevant sim_item_* columns
    for dim_name, q_ids in dim2ids.items():
        sim_cols = [f"sim_item_{qid}" for qid in q_ids if f"sim_item_{qid}" in ccr_df.columns]
        if not sim_cols:
            continue
        ccr_df[f"{dim_name}_score"] = ccr_df[sim_cols].mean(axis=1)
    
    return ccr_df