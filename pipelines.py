from typing import Callable, Dict, Any
import os
import json
import time
import pandas as pd

SYSTEM_MESSAGE = {"role": "system",
                  "content": """Analyze the following text and evaluate whether it contains each of the following elements:
    (1) causally connected events;
    (2) characters;
    (3) events that affect the characters and/or their mental states;
    (4) a plot or structure;
    (5) a normative point.
    provide a very short sentence explaining your reasoning for each element.

Return your answer as JSON,Return raw JSON only. Do not wrap the output in code blocks or additional characters of any sort:
{
  "causal_sequence": {"value": 0 or 1, "reason": "..."},
  "characters": {"value": 0 or 1, "reason": "..."},
  "internal_states": {"value": 0 or 1, "reason": "..."},
  "plot_structure": {"value": 0 or 1, "reason": "..."},
  "normative_point": {"value": 0 or 1, "reason": "..."}
}"""}


class Task:
    def __init__(self, name: str, system_message: str, parser_fn: Callable[[str], Dict[str, Any]]):
        self.name = name
        self.system_message = system_message
        self.parser = parser_fn


def parse_narrative_with_reasons(response_str):
    import json
    data = json.loads(response_str)
    result = {}
    for k, v in data.items():
        if isinstance(v, dict) and "value" in v and "reason" in v:
            result[f"{k}_value"] = v["value"]
            result[f"{k}_reason"] = v["reason"]
        else:
            result[f"{k}_value"] = None
            result[f"{k}_reason"] = None
    return result


def create_batch_requests(df, task: Task, BATCH_SIZE=50000, path='', column_name="text"):
    # OpenAI batch limit
    total_batches = (len(df) // BATCH_SIZE) + (1 if len(df) % BATCH_SIZE != 0 else 0)

    # Create output directory
    output_dir = "batch_requests"
    path = os.path.join(path, output_dir)
    os.makedirs(path, exist_ok=True)

    # Generate batch files in JSONL format
    for batch_num in range(total_batches):
        batch_start = batch_num * BATCH_SIZE
        batch_end = min((batch_num + 1) * BATCH_SIZE, len(df))
        batch_df = df.iloc[batch_start:batch_end]  # subset of the DataFrame

        # batch_filename = f"{output_dir}/batch_{batch_num+1}.jsonl"
        batch_filename = os.path.join(path, f"batch_{batch_num + 1}.jsonl")

        with open(batch_filename, "w", encoding="utf-8") as f:
            # Iterate over rows in the current batch
            for i, row in batch_df.iterrows():
                text = row[column_name]  # <-- Use the df's "text" column
                orig_index = row["orig_index"]

                custom_id = f"request-{orig_index}"  # <-- Use the df's "id" column
                max_tokens = 800  # <-- Use the df's "num_tokens" column

                # Construct the request
                request = {
                    "custom_id": str(custom_id),  # convert to string just to be safe
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            task.system_message,
                            {"role": "user", "content": text}
                        ],
                        "max_tokens": int(max_tokens)  # ensure it's an integer
                    }
                }
                # Write each JSON request object on its own line
                f.write(json.dumps(request, ensure_ascii=False) + "\n")

        print(f"Batch {batch_num + 1}/{total_batches} saved: {batch_filename}")

    print(f"✅ All {total_batches} batches created successfully in JSONL format!")


narrative_task = Task(
    name="narrative_analysis",
    system_message=SYSTEM_MESSAGE
    ,
    parser_fn=parse_narrative_with_reasons
)


def merge_response_to_df(df: pd.DataFrame, response_file: str, task) -> pd.DataFrame:
    with open(response_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # skip blank lines

            try:
                data = json.loads(line)
                custom_id = data.get("custom_id", "")
                orig_index = int(custom_id.split("-")[-1])

                content_str = data["response"]["body"]["choices"][0]["message"]["content"]
                content = json.loads(content_str)

                for key, val in content.items():
                    df.loc[df["orig_index"] == orig_index, f"{key}_value"] = val.get("value")
                    df.loc[df["orig_index"] == orig_index, f"{key}_reason"] = val.get("reason")

            except Exception as e:
                print(
                    f"⚠️ Skipping line {line_num} (custom_id: {custom_id if 'custom_id' in locals() else '?'}) due to error: {e}")

    print("✅ Merged batch responses into DataFrame")
    return df


def upload_new_batch(request_index, client, path):
    batch_path = os.path.join(path, f"batch_requests/batch_{request_index}.jsonl")
    batch_input_file = client.files.create(
        file=open(batch_path, "rb"),
        purpose="batch"
    )
    return batch_input_file


def evaluate_batch(batch_input_file, client):
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


def save_response_to_jsonl(current_batch, request_index, client):
    file_response = client.files.content(current_batch.output_file_id)

    # save to jsonl
    with open(f"batch_{request_index}_output.jsonl", "wb") as f:
        f.write(file_response.content)


def save_to_csv(df, path):
    df.to_csv(os.path.join(path, "narrative_elements.csv"), index=False)


def process_batch_output(current_batch, request_index, client, path, df, task: Task):
    """
    retrieve the completed batch, save the response to jsonl, merge the response to the dataframe, and save the dataframe to csv
    :param f: function to process the dataframe, takes in df and responses
    :param current_batch: a completed batch
    :param request_index:
    :param client:
    :param path:
    :param df:
    :return:
    """
    # retrieve the current batch to refresh the status
    current_batch = client.batches.retrieve(current_batch.id)
    # save the response to jsonl
    save_response_to_jsonl(current_batch, request_index, client)
    # merge the response to the dataframe
    merge_response_to_df(df, f"batch_{request_index}_output.jsonl", task=task)
    # save the dataframe to csv

    save_to_csv(df, path)
    # report the status
    print(f"Batch {request_index} completed and saved to CSV at {path} ✅")


def send_new_request(client, path, request_index):
    batch_input_file = upload_new_batch(request_index, client, path)
    current_batch = evaluate_batch(batch_input_file, client)
    print(f"successfully sent batch {request_index} ✅\n batch index: {current_batch.id}")

    return current_batch


def loop_batch_eval_with_queue(
        df,
        client,
        path,
        requests=[0, 12],
        delay=300,
        task: Task = narrative_task,
        folder_name=None,
        q=[],
        q_cap=1):
    out_path = os.path.join(path, "../output")
    if folder_name:
        os.makedirs(os.path.join(out_path, folder_name), exist_ok=True)
        output_path = os.path.join(out_path, folder_name)
    else:
        os.makedirs(os.path.join(out_path, "name_output"), exist_ok=True)
        output_path = os.path.join(out_path, "name_output")

    request_index = requests[0]
    current_request = send_new_request(client, path, request_index, )
    request_index += 1

    while request_index < requests[1]:
        time.sleep(delay)
        current_status = check_status(current_request, client)

        if len(q) < q_cap or current_status in ["finalizing", "completed"]:
            q.append((current_request, request_index))
            current_request = send_new_request(client, path, request_index)
            request_index += 1

        if q:
            r = q.pop(0)
            request, request_i = r
            if check_status(request, client) == "completed":
                time.sleep(delay)
                process_batch_output(request, request_i, client, output_path, df, task=task)
            else:
                q.append((request, request_i))

    ###### endgame
    while q:
        r = q.pop(0)
        request, request_i = r
        if check_status(request, client) == "completed":
            time.sleep(delay)
            process_batch_output(request, request_i, client, output_path, df, task=task)
        else:
            q.append((request, request_i))