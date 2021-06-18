from datautil import AnnDataLoader
from copy import deepcopy
import spacy
import re


class RuleBasedRelationExtractor():
    def __init__(self, nlp_model='en_core_sci_sm'):
        self.dataloader = AnnDataLoader()
        self.nlp = spacy.load(nlp_model)
        self.pattern_material = re.compile(r'<MATERIAL-(?!DESCRIPTOR)')
        self.pattern_number = re.compile(r'NUMBER')
        self.pattern_operation = re.compile(r'OPERATION')
        self.pattern_amount_unit = re.compile(r'AMOUNT-UNIT')
        self.pattern_condition_unit = re.compile(r'CONDITION-UNIT')
        self.pattern_mat_desc = re.compile(r'MATERIAL-DESCRIPTOR')
        self.pattern_condition_misc = re.compile(r'CONDITION-MISC')
        self.pattern_apparatus = re.compile(r'SYNTHESIS-APPARATUS')
        self.pattern_non_mat = re.compile(r'NONRECIPE-MATERIAL')
        self.pattern_brand = re.compile(r'BRAND')
        self.pattern_prop_misc = re.compile(r'PROPERTY-MISC')
        self.pattern_apparatus_disc = re.compile(r'APPARATUS-DESCRIPTOR')
        self.pattern_amount_misc = re.compile(r'AMOUNT-MISC')
        self.pattern_prop_type = re.compile(r'<PROPERTY-TYPE')
        self.pattern_apparatus_unit = re.compile(r'APPARATUS-UNIT')
        self.pattern_ref = re.compile(r'REFERENCE')
        self.pattern_prop_unit = re.compile(r'PROPERTY-UNIT')
        self.pattern_meta = re.compile(r'META')
        self.pattern_condition_type = re.compile(r'CONDITION-TYPE')
        self.pattern_character_apparatus = re.compile(
            r'CHARACTERIZATION-APPARATUS')
        self.pattern_apparatus_prop_type = re.compile(
            r'APPARATUS-PROPERTY-TYPE')

        self.patterns = [
            self.pattern_material, self.pattern_number, self.pattern_operation,
            self.pattern_amount_unit, self.pattern_condition_unit,
            self.pattern_mat_desc, self.pattern_condition_misc,
            self.pattern_apparatus, self.pattern_non_mat, self.pattern_brand,
            self.pattern_prop_misc, self.pattern_apparatus_disc,
            self.pattern_amount_misc, self.pattern_prop_type,
            self.pattern_apparatus_unit, self.pattern_ref,
            self.pattern_prop_unit, self.pattern_meta,
            self.pattern_condition_type, self.pattern_character_apparatus,
            self.pattern_apparatus_prop_type
        ]

        self.atm_patterns = [
            re.compile(r'air'),
            re.compile(r'argon'),
            re.compile(r'Ar'),
            re.compile(r'N2'),
            re.compile(r'nitrogen'),
            re.compile(r'H2'),
            re.compile(r'oxygen'),
            re.compile(r'O2'),
            re.compile(r'hydrogen'),
            re.compile(r'CH4'),
            re.compile(r'H2S'),
            re.compile(r'He'),
        ]

        self.solvent_patterns = [
            re.compile(r'water'),
            re.compile(r'(n|alcoh|glyc)ol'),
            re.compile(r'NaOH'),
            re.compile(r'HCl'),
            re.compile(r'acetone'),
            re.compile(r'acid'),
            re.compile(r'H2O'),
            re.compile(r'chloroform'),
            re.compile(r'sodium hydroxide'),
            re.compile(r'DMF'),
            re.compile(r'THF'),
            re.compile(r'N,N-dimethylformamide'),
            re.compile(r'hexane'),
            re.compile(r'toluene'),
            re.compile(r'H2SO4'),
            re.compile(r'EtOH'),
        ]

        self.participant_patterns = [
            re.compile(r'solution'),
            re.compile(r'mixture'),
            re.compile(r'product'),
            re.compile(r'samples'),
            re.compile(r'precipitate'),
            re.compile(r'powder'),
            re.compile(r'suspension'),
            re.compile(r'precursor'),
            re.compile(r'sample'),
            re.compile(r'products'),
            re.compile(r'chemicals'),
            re.compile(r'powders'),
            re.compile(r'GO'),
            re.compile(r'solid'),
            re.compile(r'pellets'),
            re.compile(r'solvent'),
            re.compile(r'materials'),
            re.compile(r'particles'),
            re.compile(r'material'),
            re.compile(r'gel'),
            re.compile(r'reagents'),
            re.compile(r'precursors'),
            re.compile(r'precipitates'),
            re.compile(r'carbon'),
            re.compile(r'silica'),
            re.compile(r'HCl'),
            re.compile(r'NaBH4'),
            re.compile(r'slurry'),
            re.compile(r'Cu'),
            re.compile(r'H2O'),
            re.compile(r'Samples'),
            re.compile(r'ZnO'),
            re.compile(r'KOH'),
            re.compile(r'compound'),
            re.compile(r'filtrate'),
            re.compile(r'NaOH'),
            re.compile(r'films'),
            re.compile(r'graphene'),
            re.compile(r'polymer'),
            re.compile(r'CTAB'),
            re.compile(r'zeolite'),
            re.compile(r'SnO2'),
            re.compile(r'membrane'),
            re.compile(r'hydrochloric acid')
        ]

        self.atms = {}
        self.solvs = {}
        self.precursor = {}
        self.target = {}
        self.participant = {}

    def __call__(self, *args, **kwargs):
        return self.extract(*args, **kwargs)

    def extract(self, txt_path, ann_path):
        text_raw, entities, relations_gold, event_gold = self.dataloader(
            txt_path, ann_path)
        text_masked = self.mask_entity(text_raw, entities)

        # count vocab of atmos, solvs
        # Recipe_Precursor : Operation -> Material
        # Recipe_Target : Operation -> Material
        # Participant_Material : Operation -> Material
        for tag, ev in event_gold.items():
            for e in ev:
                ent = entities[e[1]]['entity']
                if 'Atmos' in e[0]:
                    if ent not in self.atms.keys():
                        self.atms[ent] = 1
                    else:
                        self.atms[ent] += 1
                if 'Solv' in e[0]:
                    if ent not in self.solvs.keys():
                        self.solvs[ent] = 1
                    else:
                        self.solvs[ent] += 1
                if 'Precursor' in e[0]:
                    if ent not in self.precursor.keys():
                        self.precursor[ent] = 1
                    else:
                        self.precursor[ent] += 1
                if 'Target' in e[0]:
                    if ent not in self.target.keys():
                        self.target[ent] = 1
                    else:
                        self.target[ent] += 1
                if 'Participant' in e[0]:
                    if ent not in self.participant.keys():
                        self.participant[ent] = 1
                    else:
                        self.participant[ent] += 1

        # exclude first line (basename)
        text_masked = '\n'.join(text_masked.split('\n')[1:])
        text_masked = text_masked.replace('\n\n', '\n')

        doc = self.nlp(text_masked)

        sents = [self.join_tag(s) for s in doc.sents]

        # extraction start
        relations = []
        events = {}

        # [[not self.pattern_mat.search(tf) == None for tf in sent] for sent in sents]
        material = [[
            not self.pattern_material.search(tf) == None for tf in sent
        ] for sent in sents]
        number = [[not self.pattern_number.search(tf) == None for tf in sent]
                  for sent in sents]
        operation = [[
            not self.pattern_operation.search(tf) == None for tf in sent
        ] for sent in sents]
        amount_unit = [[
            not self.pattern_amount_unit.search(tf) == None for tf in sent
        ] for sent in sents]
        condition_unit = [[
            not self.pattern_condition_unit.search(tf) == None for tf in sent
        ] for sent in sents]
        mat_desc = [[
            not self.pattern_mat_desc.search(tf) == None for tf in sent
        ] for sent in sents]
        condition_misc = [[
            not self.pattern_condition_misc.search(tf) == None for tf in sent
        ] for sent in sents]
        apparatus = [[
            not self.pattern_apparatus.search(tf) == None for tf in sent
        ] for sent in sents]
        non_mat = [[
            not self.pattern_non_mat.search(tf) == None for tf in sent
        ] for sent in sents]
        brand = [[not self.pattern_brand.search(tf) == None for tf in sent]
                 for sent in sents]
        prop_misc = [[
            not self.pattern_prop_misc.search(tf) == None for tf in sent
        ] for sent in sents]
        apparatus_disc = [[
            not self.pattern_apparatus_disc.search(tf) == None for tf in sent
        ] for sent in sents]
        amount_misc = [[
            not self.pattern_amount_misc.search(tf) == None for tf in sent
        ] for sent in sents]
        prop_type = [[
            not self.pattern_prop_type.search(tf) == None for tf in sent
        ] for sent in sents]
        apparatus_unit = [[
            not self.pattern_apparatus_unit.search(tf) == None for tf in sent
        ] for sent in sents]
        ref = [[not self.pattern_ref.search(tf) == None for tf in sent]
               for sent in sents]
        prop_unit = [[
            not self.pattern_prop_unit.search(tf) == None for tf in sent
        ] for sent in sents]
        meta = [[not self.pattern_meta.search(tf) == None for tf in sent]
                for sent in sents]
        condition_type = [[
            not self.pattern_condition_type.search(tf) == None for tf in sent
        ] for sent in sents]
        character_apparatus = [[
            not self.pattern_character_apparatus.search(tf) == None
            for tf in sent
        ] for sent in sents]
        apparatus_prop_type = [[
            not self.pattern_apparatus_prop_type.search(tf) == None
            for tf in sent
        ] for sent in sents]

        mat_and_non_mat = self.merge(material, non_mat)
        synth_and_chara_apparatus = self.merge(apparatus, character_apparatus)
        mat_and_non_mat_and_apparatus = self.merge(mat_and_non_mat, apparatus)
        units = self.merge(apparatus_unit, prop_unit, condition_unit,
                           amount_unit)

        # init events
        for tag, entity in entities.items():
            if not entity['label'] == 'Operation':
                continue

            event_tag = 'E' + tag[1:]
            events[event_tag] = [('Operation', tag)]

        # Relation Extraction ##########################
        # Next_Operation : Operation -> Operation
        # Operation is connected to next Operation.
        past_ope = None
        for i, ope in enumerate(operation):
            for j, o in enumerate(ope):
                if not o:
                    continue

                head = self.get_tag(sents[i][j], tag_char='E')
                if past_ope is not None:
                    relations.append((past_ope, 'Next_Operation', head))

                past_ope = head

        final_ope = past_ope

        # ope nearest
        # for i, ope in enumerate(operation):
        #     for j, o in enumerate(ope):
        #         if not o:
        #             continue

        #         nearest = self.search_nearest(operation[i], j, direction='before')
        #         if nearest < 0:
        #             continue
        #         head = self.get_tag(sents[i][nearest],tag_char='E')
        #         tail = self.get_tag(sents[i][j],tag_char='E')
        #         relations.append((head, 'Next_Operation', tail))

        # Property_Of : Property-Misc -> Material, Nonrecipe-Material
        # connect from Property-Misc to nearest Material in same sentence.
        for i, prop in enumerate(prop_misc):
            for j, p in enumerate(prop):
                if not p:
                    continue

                nearest = self.search_nearest(mat_and_non_mat[i],
                                              j,
                                              priolity='before',
                                              direction='bidirectional')

                if nearest < 0:
                    continue

                head = self.get_tag(sents[i][j])
                tail = self.get_tag(sents[i][nearest])

                relations.append((head, 'Property_Of', tail))

        # Porperty_Of : Property-Unit -> Material
        for i, prop in enumerate(prop_unit):
            for j, p in enumerate(prop):
                if not p:
                    continue

                nearest = self.search_nearest(material[i],
                                              j,
                                              direction='before')

                if nearest < 0:
                    continue

                head = self.get_tag(sents[i][j])
                tail = self.get_tag(sents[i][nearest])

                relations.append((head, 'Property_Of', tail))

        # # Number_Of : Number -> Amount-Unit
        # # Amount-Unit conect to nearest Number before.
        # for i, amt in enumerate(amount_unit):
        #     for j, a in enumerate(amt):
        #         if not a:
        #             continue

        #         nearest = self.search_nearest(number[i], j, direction='before')
        #         if nearest < 0:
        #             continue
        #         head = self.get_tag(sents[i][nearest])
        #         tail = self.get_tag(sents[i][j])
        #         relations.append((head, 'Number_Of', tail))

        # # Number_Of : Number -> Condition-Unit
        # # Condition-Unit conect to nearest Number before.
        # for i, cond in enumerate(condition_unit):
        #     for j, c in enumerate(cond):
        #         if not c:
        #             continue

        #         nearest = self.search_nearest(number[i], j, direction='before')
        #         if nearest < 0:
        #             continue
        #         head = self.get_tag(sents[i][nearest])
        #         tail = self.get_tag(sents[i][j])
        #         relations.append((head, 'Number_Of', tail))

        # Number_Of : Number -> *-Unit

        for i, num in enumerate(number):
            for j, n in enumerate(num):
                if not n:
                    continue

                nearest = self.search_nearest(units[i], j, direction='after')

                if nearest < 0:
                    continue

                head = self.get_tag(sents[i][j])
                tail = self.get_tag(sents[i][nearest])

                relations.append((head, 'Number_Of', tail))

        # Condition_Of : Condition-Unit -> Operation
        # Condition-Unit conect to nearest Operation before
        for i, cond in enumerate(condition_unit):
            for j, c in enumerate(cond):
                if not c:
                    continue

                nearest = self.search_nearest(operation[i],
                                              j,
                                              direction='before')
                if nearest < 0:
                    continue
                head = self.get_tag(sents[i][j])
                tail = self.get_tag(sents[i][nearest], tag_char='E')
                relations.append((head, 'Condition_Of', tail))

        # Condition_Of : Condition-Misc -> Operation
        # Condition-misc conect to nearest Operation
        for i, cond in enumerate(condition_misc):
            for j, c in enumerate(cond):
                if not c:
                    continue

                nearest = self.search_nearest(operation[i],
                                              j,
                                              direction='bidirectional')
                if nearest < 0:
                    continue
                head = self.get_tag(sents[i][j])
                tail = self.get_tag(sents[i][nearest], tag_char='E')
                relations.append((head, 'Condition_Of', tail))

        # Amount_Of : Amount-Unit -> Material
        # Amount-Unit conect to nearest Material before
        for i, amt in enumerate(amount_unit):
            for j, a in enumerate(amt):
                if not a:
                    continue

                nearest = self.search_nearest(mat_and_non_mat[i],
                                              j,
                                              direction='bidirectional')
                if nearest < 0:
                    continue
                head = self.get_tag(sents[i][j])
                tail = self.get_tag(sents[i][nearest])
                relations.append((head, 'Amount_Of', tail))

        # Amount_Of : Amount-misc -> Material
        # Amount-Unit conect to nearest Material before
        for i, amt in enumerate(amount_misc):
            for j, a in enumerate(amt):
                if not a:
                    continue

                nearest = self.search_nearest(mat_and_non_mat[i],
                                              j,
                                              direction='bidirectional')
                if nearest < 0:
                    continue
                head = self.get_tag(sents[i][j])
                tail = self.get_tag(sents[i][nearest])
                relations.append((head, 'Amount_Of', tail))

        # Descriptor_Of : Material-Descriptor -> Material
        # Material-Descriptor connect to nearest Material
        for i, desc in enumerate(mat_desc):
            for j, d in enumerate(desc):
                if not d:
                    continue

                nearest = self.search_nearest(mat_and_non_mat[i],
                                              j,
                                              direction='bidirectional')
                if nearest < 0:
                    continue
                head = self.get_tag(sents[i][j])
                tail = self.get_tag(sents[i][nearest])
                relations.append((head, 'Descriptor_Of', tail))

        # Descriptor_Of : Apparutus-Descriptor -> Synthesis-Apparatus
        for i, app in enumerate(apparatus_disc):
            for j, a in enumerate(app):
                if not a:
                    continue

                nearest = self.search_nearest(apparatus[i],
                                              j,
                                              direction='bidirectional')
                if nearest < 0:
                    continue
                head = self.get_tag(sents[i][j])
                tail = self.get_tag(sents[i][nearest])
                relations.append((head, 'Descriptor_Of', tail))

        # Apparatus_Of : Synthesis-Apparatus, Characterization-Apparatus -> Operation
        for i, app in enumerate(synth_and_chara_apparatus):
            for j, a in enumerate(app):
                if not a:
                    continue

                nearest = self.search_nearest(operation[i],
                                              j,
                                              direction='before')
                if nearest < 0:
                    nearest = self.search_nearest(operation[i],
                                                  j,
                                                  direction='after')
                    if nearest < 0:
                        continue
                head = self.get_tag(sents[i][j])
                tail = self.get_tag(sents[i][nearest], tag_char='E')
                relations.append((head, 'Apparatus_Of', tail))

        # Type_Of : Property-Type -> Property-Unit
        for i, prop in enumerate(prop_type):
            for j, p in enumerate(prop):
                if not p:
                    continue

                nearest = self.search_nearest(prop_unit[i],
                                              j,
                                              direction='bidirectional')
                if nearest < 0:
                    continue
                head = self.get_tag(sents[i][j])
                tail = self.get_tag(sents[i][nearest])
                relations.append((head, 'Type_Of', tail))

        # Type_Of : Condition-Type -> Condition-Unit
        for i, cond in enumerate(condition_type):
            for j, c in enumerate(cond):
                if not c:
                    continue

                nearest = self.search_nearest(condition_unit[i],
                                              j,
                                              direction='after')
                if nearest < 0:
                    continue
                head = self.get_tag(sents[i][j])
                tail = self.get_tag(sents[i][nearest])
                relations.append((head, 'Type_Of', tail))

        # Type_Of : Apparatus-Property-Type -> Apparatus-Unit
        for i, prop_type in enumerate(apparatus_prop_type):
            for j, p in enumerate(prop_type):
                if not p:
                    continue

                nearest = self.search_nearest(apparatus_unit[i],
                                              j,
                                              direction='bidirectional')
                if nearest < 0:
                    continue
                head = self.get_tag(sents[i][j])
                tail = self.get_tag(sents[i][nearest])
                relations.append((head, 'Type_Of', tail))

        # FIXME: Brand of
        # Brand_Of : Brand -> Mat and nonrecipe-Mat
        for i, br in enumerate(brand):
            for j, b in enumerate(br):
                if not b:
                    continue

                nearest = self.search_nearest(mat_and_non_mat[i],
                                              j,
                                              direction='before')
                if nearest < 0:
                    continue
                head = self.get_tag(sents[i][j])
                tail = self.get_tag(sents[i][nearest])
                relations.append((head, 'Brand_Of', tail))

        # Brand_Of : Brand -> synthesis-apparatus
        for i, br in enumerate(brand):
            for j, b in enumerate(br):
                if not b:
                    continue

                nearest = self.search_nearest(apparatus[i],
                                              j,
                                              direction='bidirectional')
                if nearest < 0:
                    continue
                head = self.get_tag(sents[i][j])
                tail = self.get_tag(sents[i][nearest])
                relations.append((head, 'Brand_Of', tail))

        # Apparatus_Attr_Of : Apparatus-Unit -> Characterization-Apparatus, Synth-Apparatus
        for i, unit in enumerate(apparatus_unit):
            for j, u in enumerate(unit):
                if not u:
                    continue

                nearest = self.search_nearest(synth_and_chara_apparatus[i],
                                              j,
                                              direction='bidirectional')
                if nearest < 0:
                    continue
                head = self.get_tag(sents[i][j])
                tail = self.get_tag(sents[i][nearest])
                relations.append((head, 'Apparatus_Attr_Of', tail))

        # Event Extraction #########################
        # Recipe_Precursor : Operation -> Material
        # Recipe_Target : Operation -> Material
        # Participant_Material : Operation -> Material
        # cannot distinguish them without context...

        # Solvent_Material : Operation -> Material
        # Atmospheric_Material : Operaiton -> Material
        # follow dictionary
        for i, mat in enumerate(material):
            for j, m in enumerate(mat):
                if not m:
                    continue

                nearest = self.search_nearest(operation[i], j)
                if nearest < 0:
                    continue

                # decide label of relation
                label = None

                # Atmospheric_Material
                ent = entities[self.get_tag(sents[i][j])]['entity']
                for pattern in self.atm_patterns:
                    if pattern.search(ent) is not None:
                        label = 'Atmospheric_Material'

                # Solvent_Material
                if not label:
                    for pattern in self.solvent_patterns:
                        if pattern.search(ent) is not None:
                            label = 'Solvent_Material'

                # Participant_Material : Operation -> Material
                if not label:
                    for pattern in self.participant_patterns:
                        if pattern.search(ent) is not None:
                            label = 'Participant_Material'

                # Recipe_Precursor : Operation -> Material
                # Recipe_Target : Operation -> Material
                # cannot distinguish for rule
                # a number of precursor is larger so others assign to precursor
                if not label:
                    label = 'Recipe_Precursor'

                head = self.get_tag(sents[i][nearest]).replace('T', 'E')
                tail = self.get_tag(sents[i][j])
                events[head].append((label, tail))

        # Coref_Of ###########################################
        # completely same mentions are coref
        # for i, mat in enumerate(mat_and_non_mat):
        #     for j, m1 in enumerate(mat):
        #         if not m1:
        #             continue
        #         head = self.get_tag(sents[i][j])
        #         ent1 = entities[head]['entity']
        #         for k, m2 in enumerate(mat[j+1:],start=j+1):
        #             if not m2:
        #                 continue
        #             tail = self.get_tag(sents[i][k])
        #             ent2 = entities[tail]['entity']

        #             if ent1 == ent2:
        #                 relations.append((head, 'Coref_Of', tail))

        ann = self.create_ann(entities, relations, events)

        return ann

    @staticmethod
    def search_nearest(sent, pos, priolity='before',
                       direction='bidirectional'):
        res_before = False
        res_after = False
        if direction == 'bidirectional':
            dir_before = True
            dir_after = True
        elif direction == 'before':
            dir_before = True
            dir_after = False
        elif direction == 'after':
            dir_before = False
            dir_after = True

        end_before = not dir_before
        end_after = not dir_after

        for i in range(1, len(sent)):
            before = pos - i
            after = pos + i

            if dir_before and before < 0:
                end_before = True
            if dir_after and after > len(sent):
                end_after = True

            if end_before and end_after:
                return -1

            if before >= 0:
                if sent[before]:
                    res_before = True and dir_before
            if after < len(sent):
                if sent[after]:
                    res_after = True and dir_after

            if res_before and res_after:
                if priolity == 'before':
                    return before
                elif priolity == 'after':
                    return after

            if res_before:
                return before
            if res_after:
                return after
        return -1

    @staticmethod
    def get_tag(word, tag_char=None):
        if tag_char is None:
            return word.replace('>', '').split('-')[-1]
        else:
            return tag_char + word.replace('>', '').split('-')[-1][1:]

    @staticmethod
    def mask_entity(text, entities):
        convert_lists = [(ent['start'], ent['end'], '<{}-{}>'.format(
            ent['label'].upper(),
            tag,
        ), ent['entity']) for tag, ent in entities.items()]
        convert_lists_sorted = sorted(convert_lists,
                                      key=lambda x: x[1],
                                      reverse=True)

        text_conv = deepcopy(text)
        for start, end, tag, ent in convert_lists_sorted:
            assert text[start:end] == ent
            if text_conv[start - 1] != ' ':
                tag = ' ' + tag
            if len(text_conv) > end:
                if text_conv[end] != ' ':
                    tag = tag + ' '
            text_conv = text_conv[:start] + tag + text_conv[end:]

        text_conv = text_conv.replace('><', '> <')

        return text_conv

    def join_tag(self, doc):
        words = []
        tmp_word = []
        for w in doc:
            word = w.text
            if '<' in word:
                if '>' in word:
                    words.append(word)
                else:
                    tmp_word.append(word)
            elif '>' in word:
                joined_word = ''.join(tmp_word) + word
                if self.check_patterns(joined_word, self.patterns):
                    start_idx = joined_word.find('<')
                    end_idx = joined_word.find('>')
                    if end_idx + 1 == len(joined_word) and start_idx == 0:
                        words.append(joined_word)
                    else:
                        sp = [
                            joined_word[:start_idx],
                            joined_word[start_idx:end_idx + 1],
                            joined_word[end_idx + 1:]
                        ]
                        for s in sp:
                            if s:
                                words.append(s)
                else:
                    tmp_word.append(word)
                    words.extend(tmp_word)
                tmp_word = []
            else:
                if tmp_word:
                    tmp_word.append(word)
                else:
                    words.append(word)
        if tmp_word:
            words.extend(tmp_word)
        return words

    @staticmethod
    def check_patterns(word, patterns):
        result = 0
        for p in patterns:
            if p.search(word) is not None:
                result += 1

        return result

    @staticmethod
    def create_ann(entities, relations, events):
        lines = []
        # entity
        for tag, ent in entities.items():
            start = ent['start']
            end = ent['end']
            label = ent['label']
            word = ent['entity']
            line = '{}\t{} {} {}\t{}'.format(tag, label, start, end, word)
            lines.append(line)

        # relation
        r_num = 1
        for relation in relations:
            tag = 'R{}'.format(r_num)
            arg1, label, arg2 = relation
            line = '{}\t{} Arg1:{} Arg2:{}'.format(tag, label, arg1, arg2)
            lines.append(line)
            r_num += 1

        # events
        for tag, event in events.items():
            args = ' '.join(['{}:{}'.format(e[0], e[1]) for e in event])
            line = '{}\t{}'.format(tag, args)
            lines.append(line)

        ann = '\n'.join(lines)

        return ann

    @staticmethod
    def merge2(arg1, arg2):
        merged = [[False for i in range(len(a1))] for a1 in arg1]
        for i, (ar1, ar2) in enumerate(zip(arg1, arg2)):
            for j, (a1, a2) in enumerate(zip(ar1, ar2)):
                merged[i][j] = a1 or a2
        return merged

    def merge(self, *args):
        merged = [[False for i in range(len(arg))] for arg in args[0]]
        for arg in args:
            merged = self.merge2(merged, arg)

        return merged