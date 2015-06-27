#! /usr/local/bin/python3
# -*- utf-8 -*-

"""
Configurations of modeling

- feature extraction methods
- training models
- ensemble method
- score function
"""


from feature_extraction import source_event_counter, dropout_history


MODELING = {
    'features': [source_event_counter, dropout_history],
    'training_models': [],
    'ensemble_method': None,
    'score_func': 'auc'
}
