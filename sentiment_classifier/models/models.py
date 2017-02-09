""" This module redifine certain datatypes to be used  for data generation"""
import sys
import numpy
import nltk
import urllib
from numpy import random
from sentiment_classifier.pyfiles.tweets_processing import tweet_tokenize
from sentiment_classifier.pyfiles.tweets_processing import extract_pattern
from sentiment_classifier.pyfiles.tweets_processing import bag_of_words
from sentiment_classifier.pyfiles.tweets_processing import mutual_information_val

class Model(object):
    """ Parent class of all columns """

    def __init__(self, features, feature_score_type = "presence"):
        self.features = features
        self.feature_score_type = feature_score_type

    def extract_features(self,text):
        """ Given a tweet, this function extracts the value of predefined features """
        tokens = tweet_tokenize(text)
        feature_set = {}
        if self.feature_score_type == "presence":
            for feature in self.features:
                if feature in tokens:
                    feature_set[feature] = 1
                else:
                    feature_set[feature] = 0
        elif self.feature_score_type == "term_frequency":
            for feature in self.features:
                feature_set[feature] = tokens.count(feature)

        return feature_set

    def build_feature_base(self,X,Y):
        "builds feature base from a given data"
        dist_classes = list(Y.unique())
        feature_base = {}
        #Initialize empty list for each class to store features

        # Stopwords
        stoplink=urllib.urlopen("http://www.site.uottawa.ca/~diana/csi5180/StopWords")
        stopwords=stoplink.read().split("\n")

        for clss in dist_classes:
            feature_base[clss] = []

        word_count = 0
        for i in range(len(Y)):
            feature_base[Y[i]].extend(bag_of_words((X[i]).lower(),stopwords))
            word_count += len(feature_base[Y[i]])

        # Take the top 20% of bag of words as features
        for clss in feature_base.keys():
            freq_dist = nltk.FreqDist(feature_base[clss])
            Top_words = [word[0] for word in freq_dist.most_common(int(0.2*word_count))]

            feature_base[clss] = Top_words

        features = []
        for cl in feature_base.keys():
            features.extend(feature_base[cl])
        features.sort()

        return set(features)

    def evaluate_features(self,X,Y):
        """ Evaluate extracted features using mutual information """
        feature_set = self.features
        feature_class = {}
        for i in range(len(Y)):
            tokens = tweet_tokenize(X[i].lower())
            for tok in tokens:
                if tok in feature_set:
                    if feature_class.has_key(tok):
                        if feature_class[tok].has_key(Y[i]):
                            feature_class[tok][Y[i]] += 1
                        else:
                            feature_class[tok][Y[i]] = 1
                    else:
                        feature_class[tok] = {}
                        feature_class[tok][Y[i]] = 1
        class_counts = {}
        for clss in Y.unique():
            class_counts[clss] = list(Y).count(clss)

        mi = {}
        for feature in feature_class.keys():
            mi[feature] = mutual_information_val(feature_class[feature],class_counts)
        return mi

    def gen_feature_csv(self,X,Y,out_file):
        """ Generates a csv of feature vectors """

        feature_base = self.build_feature_base(X,Y)
        self.features = feature_base

        out_f = open("tests/out_data/"+out_file,"w")
        header = "class"+","+",".join(feature_base)+"\n"
        out_f.write(header)
        for i in range(len(Y)):
            feature_vector = self.extract_features(X[i].lower())
            if (feature_vector.values()).count(1) >=1:
                out_str = Y[i] + ","
                for feature in self.features:
                 out_str += str(feature_vector[feature]) + ","
                out_f.write(out_str+"\n")




    # def classify(self, feature_values):
    #     """ This function predict the sentiment of a given tweet """
    #
    # def train(self,data):
    #     """ This function trains a classifer given a set of data """
    #
