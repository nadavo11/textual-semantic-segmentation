import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import time

def create_histogram(df, column_name):
    # Create the histogram
    plt.hist(df[column_name], bins=100)  # you can adjust 'bins' as needed
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.title('Histogram of Text Lengths')
    plt.show()

def trim(df,column, interval):
    df = df[df[column] > interval[0]]
    df = df[df[column] < interval[1]]
    return df


def create_batch_requests(df,BATCH_SIZE = 50000,path=''):
    # OpenAI batch limit
    total_batches = (len(df) // BATCH_SIZE) + (1 if len(df) % BATCH_SIZE != 0 else 0)

    # Define system message
    system_message = {
        "role": "system",
        "content": "You are a highly precise and reliable assistant. Your task is to extract and retain only the core 'About Us' content from the provided scraped text. Remove any extraneous elements such as headings, promotions, or button text that may appear at the beginning or end. If no clear 'About Us' section exists, return an empty response rather than generating one. do not add anything that did not exist in the text before. If the text is not in English, translate it into English. Output only the cleaned and, if necessary, translated 'About Us' content, with no additional comments or formatting."
    }
    # Create output directory
    output_dir = "batch_requests"
    path = os.path.join(path,output_dir)
    os.makedirs(path, exist_ok=True)

    # Generate batch files in JSONL format
    for batch_num in range(total_batches):
        batch_start = batch_num * BATCH_SIZE
        batch_end = min((batch_num + 1) * BATCH_SIZE, len(df))
        batch_df = df.iloc[batch_start:batch_end]  # subset of the DataFrame

        #batch_filename = f"{output_dir}/batch_{batch_num+1}.jsonl"
        batch_filename = os.path.join(path, f"batch_{batch_num+1}.jsonl")

        with open(batch_filename, "w", encoding="utf-8") as f:
            # Iterate over rows in the current batch
            for i, row in batch_df.iterrows():
                text = row["text"]
                orig_index = row["orig_index"]

                custom_id = f"request-{orig_index}"  # <-- Use the df's "id" column
                max_tokens = row["num_tokens"]  # <-- Use the df's "num_tokens" column


                # Construct the request
                request = {
                    "custom_id": str(custom_id),   # convert to string just to be safe
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            system_message,
                            {"role": "user", "content": text}
                        ],
                        "max_tokens": int(max_tokens)  # ensure it's an integer
                    }
                }
                # Write each JSON request object on its own line
                f.write(json.dumps(request, ensure_ascii=False) + "\n")

        print(f"Batch {batch_num+1}/{total_batches} saved: {batch_filename}")

    print(f"✅ All {total_batches} batches created successfully in JSONL format!")
import pandas as pd
import json
def merge_response_to_df(df, response_file):
    # Load responses from JSONL file    
    responses = {}

    # Read and process the JSONL file
    with open(response_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())  # Ensure clean JSON parsing
                custom_id = data.get("custom_id", "")
                orig_id = int(custom_id.split("-")[-1])  # Extract orig_index from custom_id
                
                # Extract response text safely
                response_body = data.get("response", {}).get("body", {})
                choices = response_body.get("choices", [])
                if choices:
                    response_text = choices[0].get("message", {}).get("content", "").strip()
                else:
                    response_text = ""  # Default to empty string if no valid response
                
                responses[orig_id] = response_text
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line: {line[:100]}... Error: {e}")

    # Map extracted responses to the DataFrame
    df.loc[df["text_clean"].isna(), "text_clean"] = df["orig_index"].map(responses)


    # Save the updated DataFrame
    df.to_csv("updated_dataframe.csv", index=False)

    print("✅DataFrame updated successfully!")

def upload_new_batch(request_index,client):

    batch_input_file = client.files.create(
        file=open(f"batch_requests/batch_{request_index}.jsonl","rb"),
        purpose="batch"
    )
    return batch_input_file

def evaluate_batch(batch_input_file,client):
    current_batch = client.batches.create(
        input_file_id=batch_input_file.id,  # or whatever your variable name is
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )
    return current_batch  

def check_status(current_batch, client):
      current_batch = client.batches.retrieve(current_batch.id)
      return current_batch.status

def save_response_to_jsonl(current_batch,request_index,client):
  file_response = client.files.content(current_batch.output_file_id)

  #save to jsonl
  with open(f"batch_{request_index}_output.jsonl", "wb") as f:
      f.write(file_response.content)
  

def save_to_csv(df,path):
  df.to_csv(os.path.join(path,"../output/cleaned.csv"), index=False)

def loop_batch_clean(df,client,path,requests = [0,12]):
    for request_index in range(requests[0],requests[1]):
        batch_input_file = upload_new_batch(request_index,client)
        current_batch = evaluate_batch(batch_input_file,client)
        while check_status(current_batch,client) != "completed":
            time.sleep(300)
        save_response_to_jsonl(current_batch,request_index,client)
        merge_response_to_df(df,f"batch_{request_index}_output.jsonl")
        save_to_csv(df,path)



