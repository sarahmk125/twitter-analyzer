import pandas as pd
import json
import numpy as np


"""
Sources:
Loading JSONL: https://medium.com/@galea/how-to-love-jsonl-using-json-line-format-in-your-workflow-b6884f65175b
"""


def _load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data


def _jsonl_to_df(filename):
    jsonl_file = 'app/tweets/' + filename + '.jsonl'
    tweets_data = _load_jsonl(jsonl_file)
    db_data = []
    # Note: ignoring hashtags for now since they're nested
    db_cols = ['search_query', 'id_str', 'full_text', 'created_at', 'favorite_count', 'username', 'user_description']
    for d in tweets_data:
        db_data.append([])
        for col in db_cols:
            db_data[-1].append(d.get(col, float('nan')))

    tweets_df = pd.DataFrame(db_data, columns=db_cols)
    return tweets_df


def _user_grouper( filename):
    # Used in 2nd assignment
    # For each unique user, join all tweets into one tweet row in the new df.
    tweets_df = _jsonl_to_df(filename)
    users = list(tweets_df['username'].unique())
    tweets_by_user_df = pd.DataFrame(columns=['username', 'user_description', 'tweets'])

    # Iterate through all users.
    for i, user in enumerate(users):
        trunc_df = tweets_df[tweets_df['username'] == user]
        user_description = trunc_df['user_description'].tolist()[0]
        string = ' '.join(trunc_df["full_text"])
        tweets_by_user_df = tweets_by_user_df.append({'username': user, 'user_description': user_description, 'tweets': string}, ignore_index=True)

    # Return the data frame with one row per user, tweets concatenated into one string.
    return tweets_by_user_df


def save_users(filename, users_filename):
    # Load the tweets
    users_df = _user_grouper(filename)
    users_filename = 'app/tweets/' + users_filename + '.jsonl'

    with open(users_filename, 'a+') as outfile:
        for i, row in users_df.iterrows():
            # Get only related status fields
            status_new = {}

            # Add space to manually tag class
            status_new.update({'class': ''})
            # The above was hand coded, to: Is this user a Trump supporter?
            # Y = Yes, N = No, P = Political person (unclear), U = unclear, nonpolitical

            status_new.update({'user_description': row['user_description']})
            status_new.update({'username': row['username']})

            json.dump(status_new, outfile)
            outfile.write('\n')
        outfile.close()


if __name__ == "__main__":
    save_users('tweets', 'users')
