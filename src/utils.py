import os, sys
import re
import json
import glob
import datetime
from collections import Counter

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from wordcloud import WordCloud

def parse_slack_reaction(path, channel):
    """get reactions"""
    dfall_reaction = pd.DataFrame()
    combined = []
    for json_file in glob.glob(f"{path}*.json"):
        with open(json_file, 'r') as slack_data:
            combined.append(slack_data)

    reaction_name, reaction_count, reaction_users, msg, user_id = [], [], [], [], []

    for k in combined:
        slack_data = json.load(open(k.name, 'r', encoding="utf-8"))
        
        for i_count, i in enumerate(slack_data):
            if 'reactions' in i.keys():
                for j in range(len(i['reactions'])):
                    msg.append(i['text'])
                    user_id.append(i['user'])
                    reaction_name.append(i['reactions'][j]['name'])
                    reaction_count.append(i['reactions'][j]['count'])
                    reaction_users.append(",".join(i['reactions'][j]['users']))
                
    data_reaction = zip(reaction_name, reaction_count, reaction_users, msg, user_id)
    columns_reaction = ['reaction_name', 'reaction_count', 'reaction_users_count', 'message', 'user_id']
    df_reaction = pd.DataFrame(data=data_reaction, columns=columns_reaction)
    df_reaction['channel'] = channel
    return df_reaction

def get_community_participation(path):
    """ specify path to get json files"""
    combined = []
    comm_dict = {}
    for json_file in glob.glob(f"{path}*.json"):
        with open(json_file, 'r') as slack_data:
            combined.append(slack_data)
    # print(f"Total json files is {len(combined)}")
    for i in combined:
        a = json.load(open(i.name, 'r', encoding='utf-8'))

        for msg in a:
            if 'replies' in msg.keys():
                for i in msg['replies']:
                    comm_dict[i['user']] = comm_dict.get(i['user'], 0)+1
    return comm_dict


def convert_2_timestamp(column, data):
    """convert from unix time to readable timestamp
        args: column: columns that needs to be converted to timestamp
                data: data that has the specified column
    """
    if column in data.columns.values:
        timestamp_ = []
        for time_unix in data[column]:
            if time_unix == 0:
                timestamp_.append(0)
            else:
                a = datetime.datetime.fromtimestamp(float(time_unix))
                timestamp_.append(a.strftime('%Y-%m-%d %H:%M:%S'))
        return timestamp_
    else: 
        print(f"{column} not in data")

def get_tagged_users(df):
    """get all @ in the messages"""

    return df['msg_content'].map(lambda x: re.findall(r'@U\w+', x))


    
def map_userid_2_realname(user_profile: dict, comm_dict: dict, plot=False):
    """
    map slack_id to realnames
    user_profile: a dictionary that contains users info such as real_names
    comm_dict: a dictionary that contains slack_id and total_message sent by that slack_id
    """
    user_dict = {} # to store the id
    real_name = [] # to store the real name
    ac_comm_dict = {} # to store the mapping
    count = 0
    # collect all the real names
    for i in range(len(user_profile['profile'])):
        real_name.append(dict(user_profile['profile'])[i]['real_name'])

    # loop the slack ids
    for i in user_profile['id']:
        user_dict[i] = real_name[count]
        count += 1

    # to store mapping
    for i in comm_dict:
        if i in user_dict:
            ac_comm_dict[user_dict[i]] = comm_dict[i]

    ac_comm_dict = pd.DataFrame(data= zip(ac_comm_dict.keys(), ac_comm_dict.values()),
    columns=['LearnerName', '# of Msg sent in Threads']).sort_values(by='# of Msg sent in Threads', ascending=False)
    
    if plot:
        ac_comm_dict.plot.bar(figsize=(15, 7.5), x='LearnerName', y='# of Msg sent in Threads')
        plt.title('Student based on Message sent in thread', size=20)
        
    return ac_comm_dict


def get_top_20_user(data, channel='Random'):
    """get user with the highest number of message sent to any channel"""

    data['sender_name'].value_counts()[:20].plot.bar(figsize=(15, 7.5))
    plt.title(f'Top 20 Message Senders in #{channel} channels', size=15, fontweight='bold')
    plt.xlabel("Sender Name", size=18); plt.ylabel("Frequency", size=14);
    plt.xticks(size=12); plt.yticks(size=12);
    plt.show()

    data['sender_name'].value_counts()[-10:].plot.bar(figsize=(15, 7.5))
    plt.title(f'Bottom 10 Message Senders in #{channel} channels', size=15, fontweight='bold')
    plt.xlabel("Sender Name", size=18); plt.ylabel("Frequency", size=14);
    plt.xticks(size=12); plt.yticks(size=12);
    plt.show()
    
def draw_avg_reply_count(data, channel='Random'):
    """who commands many reply?"""

    data.groupby('sender_name')['reply_count'].mean().sort_values(ascending=False)[:20]\
        .plot(kind='bar', figsize=(15,7.5));
    plt.title(f'Average Number of reply count per Sender in #{channel}', size=20, fontweight='bold')
    plt.xlabel("Sender Name", size=18); plt.ylabel("Frequency", size=18);
    plt.xticks(size=14); plt.yticks(size=14);
    plt.show()

def draw_avg_reply_users_count(data, channel='Random'):
    """who commands many user reply?"""

    data.groupby('sender_name')['reply_users_count'].mean().sort_values(ascending=False)[:20].plot(kind='bar',
     figsize=(15,7.5));
    plt.title(f'Average Number of reply user count per Sender in #{channel}', size=20, fontweight='bold')
    plt.xlabel("Sender Name", size=18); plt.ylabel("Frequency", size=18);
    plt.xticks(size=14); plt.yticks(size=14);
    plt.show()

def draw_wordcloud(msg_content, week):    
    # word cloud visualization
    allWords = ' '.join([twts for twts in msg_content])
    wordCloud = WordCloud(background_color='#975429', width=500, height=300, random_state=21, max_words=500, mode='RGBA',
                            max_font_size=140, stopwords=stopwords.words('english')).generate(allWords)
    plt.figure(figsize=(15, 7.5))
    plt.imshow(wordCloud, interpolation="bilinear")
    plt.axis('off')
    plt.tight_layout()
    plt.title(f'WordCloud for {week}', size=30)
    plt.show()

def draw_user_reaction(data, channel='General'):
    data.groupby('sender_name')[['reply_count', 'reply_users_count']].sum()\
        .sort_values(by='reply_count',ascending=False)[:10].plot(kind='bar', figsize=(15, 7.5))
    plt.title(f'User with the most reaction in #{channel}', size=25);
    plt.xlabel("Sender Name", size=18); plt.ylabel("Frequency", size=18);
    plt.xticks(size=14); plt.yticks(size=14);
    plt.show()



# which user has the highest number of reply counts?
def user_with_highest_reply_counts(data):
    top_user = data.groupby('sender_name')['reply_count'].sum().idxmax()
    highest_reply_count = data.groupby('sender_name')['reply_count'].sum().max()
    
    print(f"The user with the highest number of reply counts is '{top_user}' with {highest_reply_count} replies.")


# Visualize reply counts per user per channel
def visualize_reply_counts_per_user_per_channel(data):
    avg_reply_counts = data.groupby(['channel', 'sender_name'])['reply_count'].mean().reset_index()

    unique_channels = avg_reply_counts['channel'].unique()

    for channel in unique_channels:
        channel_data = avg_reply_counts[avg_reply_counts['channel'] == channel]

        plt.figure(figsize=(12, 6))  # Set the figure size
        plt.bar(channel_data['sender_name'], channel_data['reply_count'])
        plt.title(f'Avg Reply Counts per User in #{channel}')
        plt.xlabel('Sender Name')
        plt.ylabel('Average Reply Counts')
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
        plt.tight_layout()

        plt.show()


def time_range_most_messages(data):
    # Convert 'msg_sent_time' to datetime format
    data['time_sent'] = pd.to_datetime(data['time_sent'], unit='s')

    # Extract hour from 'msg_sent_time'
    data['hour'] = data['time_sent'].dt.hour

    # Count the number of messages sent for each hour
    hour_counts = data['hour'].value_counts().sort_index()

    # Find the hour with the most messages sent
    most_active_hour = hour_counts.idxmax()
    message_count_most_active_hour = hour_counts.max()

    print(f"The hour with the most messages sent is {most_active_hour}:00 with {message_count_most_active_hour} messages.")


# what kind of messages are replied faster than others?
def messages_replied_faster(data, criterion='msg_type'):

    data['time_sent'] = pd.to_datetime(data['time_sent'], unit='s')
    data['time_thread_end'] = pd.to_datetime(data['time_thread_end'], unit='s')

   
    data['response_time'] = (data['time_thread_end'] - data['time_sent']).dt.total_seconds()

   
    avg_response_time = data.groupby(criterion)['response_time'].mean().reset_index()
    avg_response_time = avg_response_time.sort_values(by='response_time', ascending=True)

    return avg_response_time


def messages_reactions_relationship(data):

    user_interaction = data.groupby('sender_id').agg({'message_content': 'count', 'reaction_count': 'sum'}).reset_index()
    correlation = user_interaction['message_content'].corr(user_interaction['reaction_count'])
    return correlation


# Classify messages into different categories such as questions, answers, comments, etc.
def classify_messages(data):

    question_keywords = ['?', 'how', 'what', 'when', 'where', 'why', 'help']
    answer_keywords = ['answer:', 'solution:', 'reply:', 'here is', 'try this']
    comment_keywords = ['comment:', 'thoughts:', 'opinion:', 'agree', 'disagree']

    categories = []
    for message in data['message_content']:
        message_lower = message.lower()

        if any(keyword in message_lower for keyword in question_keywords):
            categories.append('Question')
        elif any(keyword in message_lower for keyword in answer_keywords):
            categories.append('Answer')
        elif any(keyword in message_lower for keyword in comment_keywords):
            categories.append('Comment')
        else:
            categories.append('Other')

    data['Message Category'] = categories

    return data
