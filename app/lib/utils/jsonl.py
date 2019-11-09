import json
import os
import pandas as pd


def _load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('[WordTokenizer] Loaded {} records from {}'.format(len(data), input_path))
    return data

def jsonl_to_df(filename, db_cols):
    jsonl_file = 'app/tweets/' + filename + '.jsonl'
    tweets_data = _load_jsonl(jsonl_file)
    db_data = []
    # Note: ignoring hashtags for now since they're nested
    for d in tweets_data:
        db_data.append([])
        for col in db_cols:
            db_data[-1].append(d.get(col, float('nan')))

    tweets_df = pd.DataFrame(db_data, columns=db_cols)
    return tweets_df