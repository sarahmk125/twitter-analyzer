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
    print('[LoadJsonl] Loaded {} records from {}'.format(len(data), input_path))
    return data


def jsonl_to_df(filename, db_cols=None):
    jsonl_file = 'app/tweets/' + filename + '.jsonl'
    tweets_data = _load_jsonl(jsonl_file)
    db_data = []

    if not db_cols:
        db_cols = [k for k in tweets_data[0].keys()]

    # Note: ignoring hashtags for now since they're nested
    for d in tweets_data:
        db_data.append([])
        for col in db_cols:
            db_data[-1].append(d.get(col, float('nan')))

    tweets_df = pd.DataFrame(db_data, columns=db_cols)
    return tweets_df


def build_filename(filename):
    dirpath = os.getcwd()
    jsonl_file = dirpath + '/app/tweets/' + filename
    if '.jsonl' not in jsonl_file:
        jsonl_file = jsonl_file + '.jsonl'
    return jsonl_file


def df_to_jsonl(df, filename):
    jsonl_file = build_filename(filename)

    # List of lists
    lol = df.values.tolist()
    keys = df.columns.tolist()

    # Write each row
    with open(jsonl_file, 'a+') as outfile:
        for row in lol:
            row_new = {}
            for i, key in enumerate(keys):
                row_new.update({key: row[i]})

            json.dump(row_new, outfile)
            outfile.write('\n')
        outfile.close()
