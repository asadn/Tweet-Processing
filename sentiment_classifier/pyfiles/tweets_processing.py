from nltk.tokenize import TweetTokenizer
import nltk
import re

def tweet_tokenize(tweet):
    tknzr = TweetTokenizer()
    return tknzr.tokenize(tweet)

def bag_of_words(tweet,stopwords):
    tweet = tweet.replace("'","")
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)
    word_text = tweet_tokenize(tweet)
    filtered_words = [word for word in word_text if word not in stopwords]
    filtered_words_2 = [word for word in filtered_words if word not in urls]
    return filtered_words_2


def extract_pattern(tweet,pattern):
    word_text = tweet_tokenize(tweet)
    tagged = nltk.pos_tag(word_text)
    NPChunker = nltk.RegexpParser(pattern)
    result = NPChunker.parse(tagged)
    phrases = []
    p = ''
    for subtree in result.subtrees(filter=lambda t: t.label() == 'NP'):
        A = subtree.leaves()
        p = ''
        for  item in A:
            p += item[0] + " "
        phrases.append(p)
    return phrases


def mutual_information_val(feature_class_counts,class_counts):
    mi = 0
    total = sum(class_counts.values())
    feature_count = sum(feature_class_counts.values())
    for cl in class_counts.keys():
        p_f_nd_c = (float(feature_class_counts[cl])/total)
        p_f = ((float(feature_count)/(total)))
        p_c = (float(class_counts[cl])/total)
        mi += (p_f_nd_c)*math.log(p_f_nd_c/((p_f)*(p_c)))
    return mi
