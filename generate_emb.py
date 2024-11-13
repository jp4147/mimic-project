import numpy as np
from tqdm import tqdm
import pickle
from itertools import islice
from openai import OpenAI

class Generate_Embeddings:
    def __init__(self, data_type = 'lab', api_key = None):

        self.client = OpenAI(api_key=api_key)
        self.data_type = data_type

        if data_type == 'lab':
            with open('lab_rev.pickle', 'rb') as h:
                self.data = pickle.load(h)
            # test = dict(islice(self.data.items(), 10))
            self.emb = self.generate_embeddings_lab(self.data)
        elif data_type == 'diag':
            with open('diag_adm_by_subject.pkl', 'rb') as h:
                self.data = pickle.load(h)            
            self.emb = self.generate_embeddings_diag(self.data)

        with open(data_type+'_emb.pickle', 'wb') as h:
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

    def generate_embeddings_lab(self, data):
        data_emb = {}

        for id, labs in tqdm(data.items()):
            if id[1] == 'demographics':
                # Process 'demographics' entries individually
                text = self.sequence_to_text(labs)
                data_emb[id] = self.get_gpt_embeddings_batch(text)
            else:
                # Process in batches
                batch_texts = [self.sequence_to_text(lab_values) for lab_values in labs.values()]
                lab_names = list(labs.keys())
                embeddings = self.get_gpt_embeddings_batch(batch_texts)

                # Create a dictionary for the embeddings of the current ID
                lab_emb = {lab_name: emb for lab_name, emb in zip(lab_names, embeddings)}
                data_emb[id] = lab_emb
        if len(data_emb)%50000==0:
            print('saving emb')
            with open(self.data_type+'_emb.pickle', 'wb') as h:
                pickle.dump(data_emb, h)            
        return data_emb

    def generate_embeddings_diag(self, data, batch_size=32):
        data_emb = {}
        batch_texts = []
        batch_ids = []
        for id, seq_label in tqdm(data.items()):
            text = self.sequence_to_text(seq_label['condition'])
            batch_texts.append(text)
            batch_ids.append(id)

            # Once batch is filled, make a single API request
            if len(batch_texts) == batch_size:
                embeddings = self.get_gpt_embeddings_batch(batch_texts)
                for i, emb in enumerate(embeddings):
                    data_emb[batch_ids[i]] = {}
                    data_emb[batch_ids[i]]['emb'] = emb
                    data_emb[batch_ids[i]]['los'] = data[batch_ids[i]]['LOS']
                # Clear the batch
                batch_texts.clear()
                batch_ids.clear()

        # Handle the last batch
        if batch_texts:
            embeddings = self.get_gpt_embeddings_batch(batch_texts)
            for i, emb in enumerate(embeddings):
                data_emb[batch_ids[i]] = {}
                data_emb[batch_ids[i]]['emb'] = emb
                data_emb[batch_ids[i]]['los'] = data[batch_ids[i]]['LOS']

        return data_emb