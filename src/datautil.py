import os
import sys
from collections import OrderedDict
import torch


# ann data loader
class AnnDataLoader():
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.load_ann(*args, **kwargs)

    def load_ann(self, txt_path, ann_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        with open(ann_path, 'r', encoding='utf-8') as f:
            ann = f.read()
        ents, rels, events = self.parse_data(text, ann)
        return text, ents, rels, events

    @staticmethod
    def parse_data(txt_raw, ann_raw):
        rels = OrderedDict()
        ents = OrderedDict()
        events = OrderedDict()
        for line in ann_raw.strip().split('\n'):
            sp = line.split('\t')
            tag = sp[0]
            if 'T' in tag:
                sp_s = sp[1].split(' ')
                label = sp_s[0]
                start = int(sp_s[1])
                end = int(sp_s[2])
                entity = sp[-1]
                ents[tag] = {
                    'label': label,
                    'start': start,
                    'end': end,
                    'entity': entity
                }
                assert txt_raw[start:
                               end] == entity, 'Not matched: span and word'
            elif 'R' in tag:
                # relation
                sp_s = sp[1].split(' ')
                # delete Operatin"-a"
                label = sp_s[0].split('-')[0]
                arg1 = sp_s[1][5:]
                arg2 = sp_s[2][5:]
                # if 'Coref_Of' in label:
                #     corefs[tag] = {
                #         'label': label,
                #         'arg1': arg1,
                #         'arg2': arg2,
                #     }
                # else:
                rels[tag] = {
                    'label': label,
                    'arg1': arg1,
                    'arg2': arg2,
                }
            elif 'E' in tag:
                if tag not in events.keys():
                    events[tag] = []
                for s in sp[1].split(' '):
                    if 'T' in s:
                        events[tag].append(s.split(':'))
            else:
                pass
        return ents, rels, events


class EntityConverter():
    def __init__(self, gold_entities, gold_events, pred_entities, pred_events):
        self.set_convert_list(gold_entities, gold_events, pred_entities,
                              pred_events)

    def set_convert_list(self, gold_entities, gold_events, pred_entities,
                         pred_events):
        # gold to pred
        self.gold_pred = self.get_convert(gold_entities, gold_events,
                                          pred_entities, pred_events)
        self.pred_gold = self.get_convert(pred_entities, pred_events,
                                          gold_entities, gold_events)

    def get_convert(self, entities1, events1, entities2, events2):
        convert = {}
        # conversion of entity to entity
        for tag1, val1 in entities1.items():
            if not tag1 in convert.keys():
                convert[tag1] = []
            start1 = val1['start']
            end1 = val1['end']
            for tag2, val2 in entities2.items():
                start2 = val2['start']
                end2 = val2['end']

                if start1 == start2 and end1 == end2:
                    if tag1 in convert.keys():
                        convert[tag1].append(tag2)

        # conversion of event to event
        for tag1, val1 in events1.items():
            if not tag1 in convert.keys():
                convert[tag1] = []
            opes = [v[1] for v in val1 if v[0] == 'Operation']
            assert len(opes) == 1
            for o in opes:
                for ent2 in convert[o]:
                    for tag2, val2 in events2.items():
                        opes2 = [v[1] for v in val2 if v[0] == 'Operation']
                        if ent2 in opes2:
                            convert[tag1].append(tag2)

        return convert

    @staticmethod
    def convert(tag, convert_list):
        return convert_list[tag]

    def g2p(self, tag):
        return self.convert(tag, self.gold_pred)

    def p2g(self, tag):
        return self.convert(tag, self.pred_gold)
