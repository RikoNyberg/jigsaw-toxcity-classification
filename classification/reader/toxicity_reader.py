# -*- coding: utf-8 -*-
# Example from https://jbarrow.ai/allennlp-the-hard-way-1/

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.instance import Instance
from overrides import overrides
from allennlp.data.tokenizers import Token
from allennlp.data.fields import Field, TextField, LabelField

import csv
import pandas as pd
import numpy as np
from typing import Dict, List, Iterator

@DatasetReader.register("toxicity_data_reader")
class ToxcityDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        def clean_special_chars(text):
            '''Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution'''
            punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
            for p in punct:
                text = text.replace(p, ' ')
            return text

        with open(file_path, 'r') as conll_file:
            datareader = csv.reader(conll_file, delimiter=',', quotechar='"')
            next(datareader)  # skip the header row
            for line in datareader:
                comment_text = line[3]
                comment_text = clean_special_chars(comment_text) # This might be useless
                target = 1 if float(line[2]) >= 0.5 else 0
                yield self.text_to_instance(comment_text, target)
        

        # def preprocess(data):
            #     '''
            #     Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
            #     '''
            #     punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
            #     def clean_special_chars(text, punct):
            #         for p in punct:
            #             text = text.replace(p, ' ')
            #         return text

            #     data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
            #     return data

            # train = pd.read_csv(file_path)
            # x_train = preprocess(train['comment_text'])
            # y_train = np.where(train['target'] >= 0.5, 1, 0)
            # # y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]

            # comment_texts = x_train.tolist()
            # targets = y_train.tolist()
            # del train
            # for comment_text, target in zip(comment_texts, targets):
            #     yield self.text_to_instance(comment_text, target)

    @overrides
    def text_to_instance(self,
                         comment_text: str,
                         target: int) -> Instance:
        fields: Dict[str, Field] = {}
        # wrap each token in the file with a token object
        tokens = TextField([Token(word) for word in comment_text.split()], self._token_indexers)

        # Instances in AllenNLP are created using Python dictionaries,
        # which map the token key to the Field type
        fields["tokens"] = tokens
        fields["label"] = LabelField(str(target))

        return Instance(fields)




