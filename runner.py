from app.lib.twitter_search import TwitterSearch

if __name__ == "__main__":
    # Perform relevant twitter searches
    TwitterSearch().search(
        query_list=[
            'violence',
            'guns',
            'mass shooting',
            'progun',
            'gun control',
            'second amendment'
        ],
        count=1000
    )

    # Visualization
