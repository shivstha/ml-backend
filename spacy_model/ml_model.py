import spacy
from itertools import groupby
from label_studio_ml.model import LabelStudioMLBase


class JapaneseNER(LabelStudioMLBase):

    def __init__(self, **kwargs):
        # don't forget to initialize base class...
        super(JapaneseNER, self).__init__(**kwargs)
        self.from_name, self.info = list(self.parsed_label_config.items())[0]
        self.to_name = self.info['to_name'][0]
        self.value = self.info['inputs'][0]['value']
        self.model = spacy.load("ja_core_news_md")


    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            input_text = task['data'].get(self.value) or task['data'].get('$Text')
            doc = self.model(input_text)
            tokens = [(tok.text, tok.idx, tok.ent_type_) for tok in doc]
            print(tokens)
            results= []
            for entity, group in groupby(tokens, key=lambda t: t[-1]):
                if not entity:
                    continue
                group = list(group)
                _, start, _ = group[0]
                word, last, _ = group[-1]
                text = ' '.join(item[0] for item in group)
                end = last + len(word)
                print(f'{text} <---> {start} <---> {end} <----> {entity}')
                print('*********************************')

                results.append({
                    'from_name': 'label',
                    'to_name': 'text',
                    'type': 'labels',
                    'value': {
                        'start': start,
                        'end': end,
                        'text': text,
                        'labels': [entity]
                    }       
                })
            predictions.append({
                'result': results,
                })
        # print(predictions)
        return predictions

   
