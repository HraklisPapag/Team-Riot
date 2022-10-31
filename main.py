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

import os
import re
from nltk.corpus import wordnet
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from shapely.geometry import LineString
from geopandas import GeoDataFrame
from shapely.geometry import Point, Polygon
import folium

from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

######################## Create Query Functions ###########################

def list_to_or(l):
    out = ''
    for index, word in enumerate(l):
        if word == word:
            out += word + ' OR '

    out = out[:(len(out)-4)]
    return out

def create_query(query_df):
    uni_kw= list(query_df['University Key Words'])
    uni_kw_string = list_to_or(uni_kw)

    unis= list(query_df['List of Universities'])
    unis_string = list_to_or(unis)

    actions= list(query_df['Actions'])
    actions_string = list_to_or(actions)

    other= list(query_df['Other'])
    other_string = list_to_or(other)

    query = ('((' + uni_kw_string + ' OR '
            + unis_string + ') ('
            + actions_string + ' OR '
            + other_string + '))')

    query += ' -is:retweet lang:en place_country:ZA'

    return query


########################## Scraping Functions #############################

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

def mentioned_users(string):
    usernames = re.findall('@[^\s]+', string)
    if usernames == []:
        return nan
    return usernames

def hashtags(string):
    hashtags = re.findall('#[^\s]+', string)
    if hashtags == []:
        return nan
    return hashtags

from cmath import nan
import advertools as adv
def extract_emojis(string):
    list = [string]
    emoji_dict = adv.extract_emoji(list)
    emojis = emoji_dict['emoji'][0]
    if(emojis == []):
        return nan
    return emojis

def find_urls(string):
    try:
        urls = re.search("(?P<url>https?://[^\s]+)", string).group("url")
    except:
        return nan
    return urls

# Define Emoji_patterns
emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

# Define the function to implement POS tagging:
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# Define the main function to clean text in various ways:
def clean_text(text):
    
    # Apply regex expressions first before converting string to list of tokens/words:
    # 1. remove @usernames
    text = re.sub('@[^\s]+', '', text)
    
    # 2. remove URLs
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text)
    
    # 3. remove hashtags entirely i.e. #hashtags
    text = re.sub(r'#([^\s]+)', '', text)
    
    # 4. remove emojis
    text = emoji_pattern.sub(r'', text)
    
    # 5. Convert text to lowercase
    text = text.lower()
    
    # 6. tokenize text and remove punctuation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    
    # 7. remove numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    
    # 8. remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    
    # 9. remove empty tokens
    text = [t for t in text if len(t) > 0]
    
    # 10. pos tag text and lemmatize text
    pos_tags = pos_tag(text)
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    
    # 11. remove words with only one letter
    text = [t for t in text if len(t) > 1]
    
    # join all
    text = " ".join(text)
    
    return(text)

def bbox_to_coords(bbox):
    all_coords = bbox.split()
    coords = []
    for coordinate in all_coords:
        coordinate = coordinate.replace('[','')
        coordinate = coordinate.replace(']','')
        coordinate = coordinate.replace(',','')
        coord = float(coordinate)
        coords.append(coord)

    return coords

def find_centroid(coords):
    geometry = LineString([(coords[0], coords[1]),(coords[2], coords[3])])
    centroid = geometry.centroid
    return centroid

########################## Content Analysis Functions #############################

def string_substring(string, substring):
    # Lower ensures that the vocab is case insensitive
    string = string.lower()
    substring = substring.lower()

    if substring in string:
        return True
    else:
        return False

def content_analysis(vocabulary):
    list_ = []

    # Loop Through Tweets
    for iter_t, tweet in tweets.iterrows():
        sub_list = []
        #Loop Through Vocab
        for iter_g, vocab in vocabulary.iterrows():
            # Loop Through Phrases in Vocab
            for phrase in vocab['Phrases']:
                # Check if Phrase is in Text
                if string_substring(tweet['text'], phrase):
                    if vocab['Conditional_Phrases'] != vocab['Conditional_Phrases']:
                        sub_list.append(vocab['Concept'])
                        break
                    else:
                        # Loop Through Conditional Phrases in Vocab
                        for con_phrase in vocab['Conditional_Phrases']:
                            # Check to see if there are Conditional Phrases otherwise break
                            if string_substring(tweet['text'], con_phrase):
                                sub_list.append(vocab['Concept'])
                                break
                            # Check if the conditional phrase has a NOT Operator
                            elif '%not%' in con_phrase:
                                con_phrase = con_phrase[6:]
                                if not string_substring(tweet['text'], con_phrase):
                                    sub_list.append(vocab['Concept'])
                                    break
        if len(sub_list) != 0:
            list_.append(sub_list)
        else:
            list_.append(nan)
    return list_

def split_string_lower(string):
    if string == string:
        string = string.lower()
        string = string.replace("'", "")
        out = string.strip('][').split(', ')
        return out
    else:
        return nan

grievances = pd.read_excel('DATA/Vocabularies/grievances.xlsx')
grievances.drop('Unnamed: 0', axis = 1, inplace = True)
grievances['Phrases'] = grievances['Phrases'].apply(lambda x: split_string_lower(x))
grievances['Conditional_Phrases'] = grievances['Conditional_Phrases'].apply(lambda x: split_string_lower(x))

triggers = pd.read_excel('DATA/Vocabularies/trigger.xlsx')
triggers.drop('Unnamed: 0', axis = 1, inplace = True)
triggers['Phrases'] = triggers['Phrases'].apply(lambda x: split_string_lower(x))
triggers['Conditional_Phrases'] = triggers['Conditional_Phrases'].apply(lambda x: split_string_lower(x))

tactics = pd.read_excel('DATA/Vocabularies/tactic.xlsx')
tactics.drop('Unnamed: 0', axis = 1, inplace = True)
tactics['Phrases'] = tactics['Phrases'].apply(lambda x: split_string_lower(x))
tactics['Conditional_Phrases'] = tactics['Conditional_Phrases'].apply(lambda x: split_string_lower(x))

actors = pd.read_excel('DATA/Vocabularies/actors.xlsx')
actors.drop('Unnamed: 0', axis = 1, inplace = True)
actors['Phrases'] = actors['Phrases'].apply(lambda x: split_string_lower(x))
actors['Conditional_Phrases'] = actors['Conditional_Phrases'].apply(lambda x: split_string_lower(x))

locations = pd.read_excel('DATA/Vocabularies/locations.xlsx')
locations.drop('Unnamed: 0', axis = 1, inplace = True)
locations['Phrases'] = locations['Phrases'].apply(lambda x: split_string_lower(x))
locations['Conditional_Phrases'] = locations['Conditional_Phrases'].apply(lambda x: split_string_lower(x))

weapons = pd.read_excel('DATA/Vocabularies/weapons.xlsx')
weapons.drop('Unnamed: 0', axis = 1, inplace = True)
weapons['Phrases'] = weapons['Phrases'].apply(lambda x: split_string_lower(x))
weapons['Conditional_Phrases'] = weapons['Conditional_Phrases'].apply(lambda x: split_string_lower(x))

eventualities = pd.read_excel('DATA/Vocabularies/eventuality.xlsx')
eventualities.drop('Unnamed: 0', axis = 1, inplace = True)
eventualities['Phrases'] = eventualities['Phrases'].apply(lambda x: split_string_lower(x))
eventualities['Conditional_Phrases'] = eventualities['Conditional_Phrases'].apply(lambda x: split_string_lower(x))

curiosities = pd.read_excel('DATA/Vocabularies/curiosity.xlsx')
curiosities.drop('Unnamed: 0', axis = 1, inplace = True)
curiosities['Phrases'] = curiosities['Phrases'].apply(lambda x: split_string_lower(x))
curiosities['Conditional_Phrases'] = curiosities['Conditional_Phrases'].apply(lambda x: split_string_lower(x))

non_protests = pd.read_excel('DATA/Vocabularies/non_protest.xlsx')
non_protests.drop('Unnamed: 0', axis = 1, inplace = True)
non_protests['Phrases'] = non_protests['Phrases'].apply(lambda x: split_string_lower(x))
non_protests['Conditional_Phrases'] = non_protests['Conditional_Phrases'].apply(lambda x: split_string_lower(x))

def unique_list(x):
    if x == x:
        return list(set(x))
    else:
        return nan

def count_occurances(col_name, label):
    global content_table
    occurances = list(content_table['Occurances'])
    for iter_t, tweet in tweets.iterrows():
            if tweet[col_name] == tweet[col_name]:
                for content in tweet[col_name]:
                    for iter_c, concept in content_table.iterrows():
                        if concept['Concept'] == content:
                            if concept['Label'] == label:
                                occurances[iter_c] += 1
                            
    content_table = content_table.drop('Occurances', axis =1)
    content_table['Occurances'] = occurances

########################## Sentiment Analysis Functions #############################

def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # removes mentions
    text = re.sub(r'#', '', text) # removes #
    text = re.sub(r'RT[\s]+', '', text) # removes RT
    text = re.sub(r'https?:\/\/\S+', '', text) # removes hyper link

    return text

# get the subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# get the polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

# function to compute the negative, neutral and positive analysis
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

########################## University Locator Functions #############################

def split_string(string):
    if string == string:
        string = string.replace("'", "")
        out = string.strip('][').split(',')
        return out
    else:
        return nan

def string_substring_case_sensitive(string, substring):
    # Lower ensures that the vocab is case insensitive
    if substring in string:
        return True
    else:
        return False

def university_locator():
    addition = []
    for iter_t, tweet in tweets.iterrows():
        coords = []
        words = []
        for iter_u, uni in universities.iterrows():
            if uni['Nickname'] == uni['Nickname']:
                for nickname in uni['Nickname']:
                    if string_substring(tweet['text'], nickname):
                        # coords = [uni['Latitude'], uni['Longitude']]#lat, long
                        words.append(uni['Universities'])
            if uni['Abbriviation'] == uni['Abbriviation']:        
                for abbr in uni['Abbriviation']:
                    if string_substring_case_sensitive(tweet['text'], abbr):
                        words.append(uni['Universities'])
                
                if string_substring(tweet['text'], uni['Universities']):
                    words.append(uni['Universities'])
        words = list(set(words))
        if len(words) == 0:
            words = nan
        addition.append(words)

    return addition

def change_coords():
    for iter_t, tweet in tweets.iterrows():
        if tweet['universities'] == tweet['universities']:
            if len(tweet['universities']) == 1:
                new_lat = (list(universities.loc[universities['Universities'] == tweet['universities'][0], 'Latitude']))[0]
                new_long = (list(universities.loc[universities['Universities'] == tweet['universities'][0], 'Longitude']))[0]
                new_geometry = Point(new_long,new_lat)
                tweets['geometry'][iter_t] = new_geometry



################################################ MAIN METHOD ################################################

################################ Create Query ################################
query_df = pd.read_excel('DATA/Protest-Query.xlsx')
query = create_query(query_df)

################################ Scrape Tweets ################################
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAEGchQEAAAAAZSYFv1nyLDV81YAKEfDr1fVrlho%3DWKBvyLhQ4CeHrlBRtecAetYkB1ZnAjI3Zydb1516fkIzKhS4vh'

start_date = datetime.datetime(2017,10,24)
end_date = datetime.datetime(2017,10,26)

protest_tweets = input_query(query=query, start_date=start_date, end_date=end_date, client=set_client(bearer_token))
scrape_tweets = populate_df(protest_tweets=protest_tweets)
scrape_tweets.to_csv('DATA/Scrape Tweets.csv')

################################ Clean Tweets ################################

tweets = scrape_tweets

# tweets.drop_duplicates(keep = False, inplace = True)
tweets.dropna(subset = ['text'], inplace = True)

tweets['mentioned_users'] = tweets['text'].apply(lambda x: mentioned_users(x))
tweets.head()

tweets['hashtags'] = tweets['text'].apply(lambda x: hashtags(x))
tweets.head()

tweets['emojis'] = tweets['text'].apply(lambda x: extract_emojis(x))
tweets.head()

tweets['urls'] = tweets['text'].apply(lambda x: find_urls(x))
tweets.head()

tweets['cleaned_text'] = tweets['text'].apply(lambda x: clean_text(x))
tweets.head()

# tweets['coords'] = tweets['bbox'].apply(lambda x: bbox_to_coords(x))
tweets['coords'] = tweets['bbox']

tweets['longitude_1'] = tweets['coords'].apply(lambda x: x[0])
tweets['latitude_1'] = tweets['coords'].apply(lambda x: x[1])
tweets['longitude_2'] = tweets['coords'].apply(lambda x: x[2])
tweets['latitude_2'] = tweets['coords'].apply(lambda x: x[3])

tweets.drop('bbox', axis=1, inplace=True)

tweets['centroid'] = tweets['coords'].apply(lambda x: find_centroid(x))

tweets['centroid_long'] = tweets['centroid'].apply(lambda z: z.x)
tweets['centroid_lat'] = tweets['centroid'].apply(lambda z: z.y)

tweets.drop('centroid', axis =1, inplace=True)

# Edge cases
tweets.loc[tweets.place_name == 'Cape Town, South Africa', ['centroid_long', 'centroid_lat']] = 18.4241, -33.9249
tweets.loc[tweets.place_name == 'Mdumbi Beach', ['centroid_long', 'centroid_lat']] = 29.215369, -31.933896
tweets.loc[tweets.place_name == "Betty's Bay, South Africa", ['centroid_long', 'centroid_lat']] = 18.92051, -34.34747
tweets.loc[tweets.place_name == 'Bloubergstrand', ['centroid_long', 'centroid_lat']] = 18.46173, -33.800418

geometry = [Point(xy) for xy in zip(tweets['centroid_long'], tweets['centroid_lat'])]
gdf = GeoDataFrame(tweets, geometry=geometry)  

tweets.drop('longitude_1', axis=1, inplace=True)
tweets.drop('longitude_2', axis=1, inplace=True)
tweets.drop('latitude_1', axis=1, inplace=True)
tweets.drop('latitude_2', axis=1, inplace=True)
tweets.drop('centroid_long', axis=1, inplace=True)
tweets.drop('centroid_lat', axis=1, inplace=True)

tweets = tweets[tweets['place_name'] != 'South Africa']

tweets.drop('coords', axis=1, inplace=True)

tweets.to_csv('DATA/Clean Tweets.csv')

################################ Content Analysis ################################

tweets['grievances'] = content_analysis(grievances)
tweets['grievances'] = tweets['grievances'].apply(lambda x: unique_list(x))

tweets['triggers'] = content_analysis(triggers)
tweets['triggers'] = tweets['triggers'].apply(lambda x: unique_list(x))

tweets['tactics'] = content_analysis(tactics)
tweets['tactics'] = tweets['tactics'].apply(lambda x: unique_list(x))

tweets['actors'] = content_analysis(actors)
tweets['actors'] = tweets['actors'].apply(lambda x: unique_list(x))

tweets['locations'] = content_analysis(locations)
tweets['locations'] = tweets['locations'].apply(lambda x: unique_list(x))

tweets['weapons'] = content_analysis(weapons)
tweets['weapons'] = tweets['weapons'].apply(lambda x: unique_list(x))

tweets['eventualities'] = content_analysis(eventualities)
tweets['eventualities'] = tweets['eventualities'].apply(lambda x: unique_list(x))

tweets['curiosities'] = content_analysis(curiosities)
tweets['curiosities'] = tweets['curiosities'].apply(lambda x: unique_list(x))

tweets['non_protests'] = content_analysis(non_protests)
tweets['non_protests'] = tweets['non_protests'].apply(lambda x: unique_list(x))

tweets.to_csv('DATA/Tweet Content.csv')

content_table = pd.read_excel('DATA/Protest is SA   SAS rules V2 - Separated.xlsx')
content_table = content_table.drop('Phrases', axis =1)
content_table = content_table.drop('Conditional_Phrases', axis =1)
content_table = content_table.drop('Afrikaans_Phrases', axis =1)
content_table = content_table.drop('Rule', axis =1)
content_table['Occurances'] = 0

count_occurances('grievances', 'Grievance')
count_occurances('triggers', 'Trigger')
count_occurances('tactics', 'Tactic')
count_occurances('actors', 'Actors')
count_occurances('locations', 'Location')
count_occurances('weapons', 'Weapons')
count_occurances('eventualities', 'Eventuality')
count_occurances('curiosities', 'Curiosity')
count_occurances('non_protests', 'Non-protest')

content_table.to_csv('DATA/Content Table.csv')

total = len(tweets)
frequency_table = content_table.groupby('Label').Occurances.sum().reset_index()
frequency_table['Frequency'] = frequency_table['Occurances'].apply(lambda x: round((x/total)*100,2))

frequency_table.to_csv('DATA/Frequency Table.csv')


################################ Sentiment Analysis ################################

text = tweets['text'].tolist()
text = pd.DataFrame(text, columns = ['Text'])

text['Text']=text['Text'].apply(cleanTxt)

text['Subjectivity'] = text['Text'].apply(getSubjectivity)
text['Polarity'] = text['Text'].apply(getPolarity)
text['Analysis'] = text['Polarity'].apply(getAnalysis)


text.drop(['Text'], axis = 1, inplace = True)

tweets = pd.concat([tweets, text.reindex(tweets.index)], axis = 1)

tweets.to_csv('DATA/Tweet Sentiment.csv')

##################################### University Locations #####################################

universities = pd.read_excel('DATA/South African Universities.xlsx')
universities['Nickname'] = universities['Nickname'].apply(lambda x: split_string(x))
universities['Abbriviation'] = universities['Abbriviation'].apply(lambda x: split_string(x))
tweets['universities'] = university_locator()

change_coords()

tweets.to_csv('DATA/University_Locations.csv')