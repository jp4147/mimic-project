import numpy as np
from tqdm import tqdm
import pickle
from itertools import islice
from openai import OpenAI

class Generate_Embeddings:
    def __init__(self, data_type = 'demog', api_key = None):

        self.client = OpenAI(api_key=api_key)
        self.data_type = data_type

        with open('datasets/mimic_data.pickle', 'rb') as h:
            self.data = pickle.load(h)

        if data_type == 'lab':
            # test = dict(islice(self.data.items(), 10))
            id = list(self.data.keys())[0]
            lab_types = list(self.data[id][data_type].keys())
            for sub_cat in lab_types:
                self.emb = self.generate_embeddings(self.data, data_type, sub_cat = sub_cat)
                with open('datasets/'+ data_type+'_'+sub_cat+'_emb.pickle', 'wb') as h:
                    pickle.dump(self.emb, h)
        else:         
            self.emb = self.generate_embeddings(self.data, data_type)

            with open('datasets/'+ data_type+'_emb.pickle', 'wb') as h:
                pickle.dump(self.emb, h)
    
    def get_gpt_embeddings_batch(self, texts):
        # Batching the requests for embeddings
        response = self.client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [res.embedding for res in response.data]

    def sequence_to_text(self, sequence):
        if sequence:
            seq2text = " ".join(map(str, sequence))
        else:
            seq2text = "[]"
        return seq2text

    def generate_embeddings(self, data, cat, sub_cat = '', batch_size=516):
        data_emb = {}
        batch_texts = []
        batch_ids = []

        for id_adm, dat in tqdm(data.items()):
            if len(sub_cat)>0:
                text = self.sequence_to_text(dat[cat][sub_cat])
            else:
                text = self.sequence_to_text(dat[cat])
            batch_texts.append(text)
            batch_ids.append(id_adm)

            # Once batch is filled, make a single API request
            if len(batch_texts) == batch_size:
                embeddings = self.get_gpt_embeddings_batch(batch_texts)
                for i, emb in enumerate(embeddings):
                    data_emb[batch_ids[i]] = {}
                    data_emb[batch_ids[i]]['emb'] = emb
                    data_emb[batch_ids[i]]['los'] = data[batch_ids[i]]['los']
                # Clear the batch
                batch_texts.clear()
                batch_ids.clear()

        # Handle the last batch
        if batch_texts:
            embeddings = self.get_gpt_embeddings_batch(batch_texts)
            for i, emb in enumerate(embeddings):
                data_emb[batch_ids[i]] = {}
                data_emb[batch_ids[i]]['emb'] = emb
                data_emb[batch_ids[i]]['los'] = data[batch_ids[i]]['los']

        return data_emb
    
