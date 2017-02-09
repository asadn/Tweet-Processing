""" Contains different tests """
from nose.tools import assert_equal
from nose.tools import assert_raises
from sentiment_classifier.models.models import *
from datetime import datetime
from sentiment_classifier.pyfiles.general import generate_dates
import pandas as pd
import urllib

""" Test models """

class TestModels:

    def test__extract_features(self):
        """ Tests extract features function """
        text_sample = "I really really love this movie"
        feature_sample = ['really','love','good']
        feature_score_type = "presence"
        model_sample = Model(feature_sample,feature_score_type)
        result_features = model_sample.extract_features(text_sample)
        assert_equal(result_features,{'really':1,'love':1,'good':0})
        feature_score_type = "term_frequency"
        model_sample = Model(feature_sample,feature_score_type)
        result_features = model_sample.extract_features(text_sample)
        assert_equal(result_features,{'really':2,'love':1,'good':0})

    def test_build_feature_base(self):
        """ Tests the function build feature base """
        data = pd.DataFrame(pd.read_csv("tests/in_data/pro1_sub.csv"))

        X = data.ix[:,1]
        Y = data.ix[:,0]
        model_sample = Model([],"presence")

        feature_base = model_sample.build_feature_base(X,Y)
        feature_evaluation =
        assert_equal(len(feature_base) > 10, True)

    def test_gen_feature_csv(self):
        data = pd.DataFrame(pd.read_csv("tests/in_data/pro1_sub.csv"))
        out_file = "poa_pro1_sub.csv"
        X = data.ix[:,1]
        Y = data.ix[:,0]
        model_sample = Model([],"presence")
        model_sample.gen_feature_csv(X,Y,out_file)
        assert_equal(False,True)
