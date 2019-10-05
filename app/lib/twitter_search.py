import json
import os
import pandas as pd
import app.constants as constants
from twython import Twython
from math import ceil


class TwitterSearch(object):
    def __init__(self):
        pass

    def _tweets_to_jsonl(self, statuses, filename, search_query):
        with open(filename, 'a+') as outfile:
            for status in statuses:
                # Get only related status fields
                status_new = {}
                status_new.update({'search_query': search_query})
                status_new.update({'id_str': status['id_str']})
                status_new.update({'full_text': status['full_text']})
                status_new.update({'created_at': status['created_at']})
                status_new.update({'favorite_count': status['favorite_count']})
                status_new.update({'hastags': status['entities'].get('hashtags')})
                status_new.update({'username': status['user']['screen_name']})
                status_new.update({'user_description': status['user']['description']})

                json.dump(status_new, outfile)
                outfile.write('\n')
            outfile.close()

    def _single_search(self, query_string, result_type, count, filename):
        # Instantiate an object
        python_tweets = Twython(constants.TWITTER_CONSUMER_API_KEY, constants.TWITTER_CONSUMER_API_SECRET_KEY)

        # Create our query; note, count max is 100.
        query = {
            'q': query_string,
            'result_type': result_type,
            'count': count if count else 100,
            'lang': 'en',
            'tweet_mode': 'extended'
        }

        # Note: Only looking at status; in the future, media might be interesting.
        tweets = python_tweets.search(**query)
        statuses = tweets['statuses']
        print(f"[TwitterSearch] Found {len(statuses)} tweets to write for query '{query_string}'.")

        dirpath = os.getcwd()
        filename = filename if filename else 'tweets'
        jsonl_file = dirpath + '/app/tweets/' + filename
        if '.jsonl' not in jsonl_file:
            jsonl_file = jsonl_file + '.jsonl'
        self._tweets_to_jsonl(statuses, jsonl_file, query_string)
        print(f"[TwitterSearch] Wrote tweets to {filename}.")

    def _search_set(self, query_string, query_list, result_type, count, filename):
        if not query_string and not query_list:
            print("[TwitterSearch] Must provide query string or list, not searching...")
            return

        if query_string:
            print(f"[TwitterSearch] Found 1 query, executing...")
            self._single_search(query_string, result_type, count, filename)
            return

        print(f"[TwitterSearch] Found {len(query_list)} queries, executing...")
        for query in query_list:
            self._single_search(query, result_type, count, filename)

    def search(self, query_string=None, query_list=None, result_type=None, count=None, filename=None):
        if not count or count <= 100:
            self._search_set(query_string, query_list, result_type, count, filename)
            return

        # Requires each query to have more than 100, math things
        iters = ceil(count / 100)
        last_iter_count = count % 100
        for i in range(0, iters):
            # If it's the last iteration, only do last mod amount of count.
            if i == iters - 1:
                self._search_set(query_string, query_list, result_type, last_iter_count, filename)
                continue

            # Otherwise, get 100 as the count each time.
            self._search_set(query_string, query_list, result_type, count, filename)
