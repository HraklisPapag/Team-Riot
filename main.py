# Import Libraries

from tweepy import Stream
from tweepy import OAuthHandler
#from tweepy.streaming import StreamListener
import tweepy
import json
import pandas as pd
import csv
import string
import time
import datetime
import numpy as np
from cmath import nan

########################## Scraping Functions#############################

def set_client(bearer_token):
    client = tweepy.Client(bearer_token, wait_on_rate_limit=True)
    return client

def input_query(query, start_date, end_date, client):
    numTweets = 500
    protest_tweets = []
    count = 0 
    for response in tweepy.Paginator(client.search_all_tweets, 
                                    query = query,
                                    user_fields = ['public_metrics'],
                                    tweet_fields = ['created_at', 'geo', 'public_metrics', 'text'],
                                    place_fields = ['id', 'geo', 'name', 'country_code', 'place_type', 'full_name', 'country'],
                                    expansions = ['author_id', 'geo.place_id'],
                                    start_time = start_date,
                                    end_time = end_date,
                                    max_results=numTweets
                                ):
        time.sleep(1)
        protest_tweets.append(response)

    return protest_tweets

def populate_df(protest_tweets):
    error_count = 0
    result = []
    user_dict = {}
    place_dict = {}
    # Loop through each response object
    for response in protest_tweets:
        # Take all of the users, and put them into a dictionary of dictionaries with the info we want to keep
        for user in response.includes['users']:
            user_dict[user.id] = {'followers': user.public_metrics['followers_count'],
                                'tweets': user.public_metrics['tweet_count'],
                                }
        
        place_dict = {p['id']: p for p in response.includes['places']}

        for tweet in response.data:
            # For each tweet, find the author's information
            author_info = user_dict[tweet.author_id]
            # Put all of the information we want to keep in a single dictionary for each tweet

            try:
                if place_dict[tweet.geo['place_id']]:
                    place = place_dict[tweet.geo['place_id']]   
                else:
                    place = nan

                result.append({'tweet_id': tweet.id,
                        'author_followers': author_info['followers'],
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'retweets': tweet.public_metrics['retweet_count'],
                        'replies': tweet.public_metrics['reply_count'],
                        'likes': tweet.public_metrics['like_count'],
                        'quote_count': tweet.public_metrics['quote_count'],
                        'place_id': tweet.geo,
                        'place_name': place.full_name,
                        'bbox': place.geo['bbox']

                        })

            except:
                error_count = error_count + 1


    print(error_count, " result(s) were excluded due to invalid data points.")        

    # Change this list of dictionaries into a dataframe             
    df = pd.DataFrame(result)      
    return df


########################## Cleaning Functions#############################




################# MAIN METHOD ####################

################ Scrape Tweets####################
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAEGchQEAAAAAZSYFv1nyLDV81YAKEfDr1fVrlho%3DWKBvyLhQ4CeHrlBRtecAetYkB1ZnAjI3Zydb1516fkIzKhS4vh'



query = '((University OR Universities OR Student OR tech OR school OR lecture OR res OR campus OR education OR teacher OR lecturer OR accommodation OR Wits OR UJ OR Tuks OR UCT OR UWC OR Rhodes OR Stellies OR Maties OR UNISA OR UKZN OR NWU OR UFS OR DUT OR NMMU OR CUT) (demonstration OR protest OR riot OR strike OR march OR rally OR rallies OR burning OR barricade OR vandal OR struggle OR shutdown OR boycot OR demonstrate OR disrupt OR occupy OR throw OR burn OR block OR damage OR picket OR gathering OR dance OR sing OR toyi OR chant OR amandla OR mass%meeting OR fee OR must fall OR SRC OR registration OR free%education)) -is:retweet lang:en place_country:ZA'
start_date = datetime.datetime(2017,10,24)
end_date = datetime.datetime(2017,10,26)

protest_tweets = input_query(query=query, start_date=start_date, end_date=end_date, client=set_client(bearer_token))
scrape_tweets = populate_df(protest_tweets=protest_tweets)
print('Scraped Dataframe Dimenesions: ', scrape_tweets.shape, '(rows,columns)')
print(scrape_tweets.head())
scrape_tweets.to_csv('DATA/scrape_tweets.csv')

################ Clean Tweets####################
