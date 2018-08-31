'''
Author: B. Aditya, aditya.bulusu168@gmail.com
Title: Emotion Detection in tweets
Description: The following code corresponds to emotion detection in tweets, classifying them into fear, neutral, weak, sad, etc.
             and generate results in the form of bar charts, word clouds and tables. The classification is done with the use of
             a Naive Bayes Classifier ( from NLTK ) fed with a training data set of 40000 tweets of labelled tweets with their
             corresponding emotions. Changing the training data set can help add more classes to the implementation.Dig deep into
             the code to understand the implementation. Tweets are obtained in windows of 100 ( for whatever the number of tweets
             are to be analyzed ).
Modules: tweepy, nltk, csv, plotly, wordcloud
'''
# All the necessary module imports
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from tweepy import OAuthHandler,API,Cursor
from os import environ
from re import sub
import matplotlib.pyplot as pplot
from wordcloud import WordCloud
from plotly.offline import plot
import plotly.graph_objs as go
from csv import reader

# Dictionary of the words obtained
all_words = set()


# Utility function for cleaning tweets
def clean_tweet(tweet):
    cleaned_tweet = sub(r'@\w+|https://\w+|www.\w+', '', tweet)  # Remove Hyperlinks and references
    cleaned_tweet = sub(r'\W+', ' ', cleaned_tweet)  # Remove unnecessary characters
    cleaned_tweet = sub(r'#', '', cleaned_tweet)  # Remove hash tags
    cleaned_tweet = sub(r'RT', '', cleaned_tweet)  # Remove the re-tweets
    return cleaned_tweet


# Utility funtion to obtain the training data set for the classifier
def get_training_data():
    print('Please wait... Building your training data set...')
    all_data = []  # List of all tweets and their labels in tuple format, initially empty
    data_file = open("text_emotion.csv")  # Training data file
    csv_reader = reader(data_file)
    tweets = 0
    threshold = 5000  # Change this to modify the number of tweets to be used as training data set ( Maximum for the data set provided : 40000 )
    for data in csv_reader:
        all_data.append((clean_tweet(data[1]), data[0]))
        tweets += 1
        if tweets > threshold:
            break
    global all_words  # Access the dictionary of words
    print('Cleaning the data set...')
    for data in all_data:  # Generating a dictionary of the all the words
        for word in word_tokenize(data[0]):
            all_words.add(word)
    training_data = []
    print('Constructing the data set...')
    for data in all_data:  # Constructing the data set as a list of dictionary where each dictionary corresponds to presence of a word in the tweet
        data_dict = {}
        for word in all_words:
            data_dict[word] = word in word_tokenize(data[0])
        training_data.append((data_dict, data[1]))
    print('Constructing data set successful!')
    return training_data


# Utility function to get a trained Naive Bayes Classifier
def get_classifier():
    training_data = get_training_data()  # Obtain the training data set
    print('Training the classifier...')
    classifier = NaiveBayesClassifier.train(training_data)  # Train the classifier for the training data set
    print('Classifier successfully trained...')
    return classifier


# Utility function to create an API from the credentials (stored as environment variables)
def setup_twitter():
    api = None
    try:
        consumer_key = environ['TWITTER_API_KEY']
        consumer_secret_token = environ['TWITTER_API_SECRET_KEY']
        access_token = environ['TWITTER_ACCESS_TOKEN']
        access_secret_token = environ['TWITTER_ACCESS_SECRET_TOKEN']
        print('Connecting to twitter...')
        auth_handle = OAuthHandler(consumer_key, consumer_secret_token)  # Authenticating the user to get a handler
        auth_handle.set_access_token(access_token, access_secret_token)  # Setting the access tokens for the api
        api = API(auth_handle, wait_on_rate_limit=True)  # Getting an API for the handler
        print('Connected! Analyzing tweets... (this might take time based on the number of tweets)')
    except Exception as e:
        print("Error occurred! Message: ",e)
    return api


# Utility function to classify the tweets
def get_tweets_and_sentiment(hashtags=None, number_of_tweets=1000):
    global all_words
    api = setup_twitter()
    if api is None:
        return "Twitter setup failed! Please check your credentials!"
    results = {}
    classifier = get_classifier()
    for hashtag in hashtags:
        print('Analysis for', hashtag, 'started.')
        emotions = {}  # Dictionary of all the emotions and no. of tweets that contain the emotion
        analyzed_tweets = 0
        last_window = number_of_tweets % 100  # Calculating the last window for the given number of tweets
        text = []
        while analyzed_tweets < number_of_tweets :
            count_per_request = 100
            if (number_of_tweets - analyzed_tweets) == last_window:
                count_per_request = last_window
            for tweet in Cursor(api.search, q=hashtag, lang='en').items(count_per_request):  # Obtaining 100 tweets at a time
                try:
                    tweet_data = clean_tweet(str(tweet.text))  # Pre-processing the tweet
                    analyzed_tweets += 1
                except Exception:
                    continue
                data_dict = {}
                for word in all_words:  # Constructing a testing data dictionary for all the words
                    data_dict[word] = word in word_tokenize(tweet_data)
                    if data_dict[word]:
                        text.append(word)
                emotion = classifier.classify(data_dict)  # Classify the tweet and store the emotion in the results
                try:
                    emotions[emotion] += 1
                except KeyError:
                    emotions[emotion] = 1
            print(str(analyzed_tweets), " tweets analyzed for", hashtag, " Remaining tweets: ", str(number_of_tweets - analyzed_tweets))
        print("Analysis for ", hashtag, "completed.")
        word_cloud = WordCloud().generate_from_text(' '.join(text))  # Generate a word cloud for a hash tag
        pplot.imsave(arr=word_cloud, fname='Word_Cloud_'+hashtag+'.png', format='png')
        print('Word Cloud for', hashtag, 'generated successfully as Word_Cloud_'+hashtag)
        results[hashtag] = emotions
    return results


# Main function to accept input and obtain bar chart and table for the result
if __name__ == '__main__':
    try:
        hashtags_set = input('Enter Hashtags (separate by comma\',\' for multiple hashtags) : ')
        hashtags = sub(r'\W&^,', '', hashtags_set)
        hashtags = hashtags.split(',')
        hashtags = [''.join(['#', hashtag]) for hashtag in hashtags]  # Getting a list of hash tags from the user
        hash_set = hashtags_set.replace(',', '_')
        number_of_tweets = int(input('Enter maximum number of tweets (minimum 1000 suggested) : '))
        results = get_tweets_and_sentiment(hashtags, number_of_tweets)
        print("Please wait while we generate the bar plot and table for the results...")
        bar_data = []
        emotions = []
        for hashtag, data in results.items():
            plot({  # Generate a bar chart for each hash tag with the corresponding emotions and count
                "data": [go.Bar(x=list(data.keys()), y=list(data.values()))],

            }, auto_open=True, image='png', image_filename="Bar_Plot_" + hashtag, filename="Bar_Plot_" + hashtag + '.html')
            plot({  # Generating table for the results
                    "data": [go.Table(
                        header=dict(
                            values=["Hashtag"] + list(data.keys()),
                            fill=dict(color='#C2D4FF'),
                            align=['left'] * 5
                        ),
                        cells=dict(
                            values=[hashtag] + list(data.values()),
                            fill=dict(color='#F5F8FF'),
                            align=['left'] * 5
                        )
                    )]
                }, auto_open=True, image='png', image_filename='Table_' + hashtag, filename='Table_' + hashtag + '.html')
        print('Bar plot successfully generated and saved as', 'Bar_Plot_' + hash_set)
        print('Table representation generated and saved as', 'Table_' + hash_set)
    except Exception as e:
        print('Error occurred(main)! Message: ', e)
