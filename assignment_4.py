import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import jsonlines
import torch
import torch.nn as nn
from torch import tensor 
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers.pipelines import TextClassificationPipeline

from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients, LayerActivation
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

class ExplainableTransformerPipeline():
    def __init__(self, name:str, pipeline: TextClassificationPipeline, device: str):
        self.__name = name
        self.__pipeline = pipeline
        self.__device = device
        self.ref_token_id = self.__pipeline.tokenizer.pad_token_id # A token used for generating token reference
        self.sep_token_id = self.__pipeline.tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
        self.cls_token_id = self.__pipeline.tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence

    def predict(self, inputs, token_type_ids=None, position_ids=None, attention_mask=None):
        output = self.__pipeline.model(inputs, token_type_ids=token_type_ids,
                    position_ids=position_ids, attention_mask=attention_mask, )
        return output.logits, output.attentions
    
    def squad_pos_forward_func(self, inputs, token_type_ids=None, position_ids=None, attention_mask=None, position=0):
        pred = self.__pipline.model(inputs_embeds=inputs, token_type_ids=token_type_ids,
                    position_ids=position_ids, attention_mask=attention_mask, )
        pred = pred[position]
        return pred.max(1).values
    
    def construct_input_ref_pair(self, text):
        # question_ids = self.__pipeline.tokenizer.encode(question, add_special_tokens=False)
        text_ids = self.__pipeline.tokenizer.encode(text, add_special_tokens=False)

        # construct input token ids
        # input_ids = [self.cls_token_id] + text_ids + [self.sep_token_id]

        # construct reference token ids 
        ref_input_ids = [self.ref_token_id] * len(text_ids)

        return torch.tensor([text_ids], device=self.__device), torch.tensor([ref_input_ids], device=self.__device)

    def construct_input_ref_token_type_pair(self, input_ids, sep_ind=0):
        seq_len = input_ids.size(1)
        token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=self.__device)
        ref_token_type_ids = torch.zeros_like(token_type_ids, device=self.__device)# * -1
        return token_type_ids, ref_token_type_ids

    def construct_input_ref_pos_id_pair(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.__device)
        # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
        ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=self.__device)

        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
        return position_ids, ref_position_ids
        
    def construct_attention_mask(self, input_ids):
        return torch.ones_like(input_ids)
    
    def visualize_token2token_scores(self, scores_mat, outfile_path, x_label_name='Head'):
        fig = plt.figure(figsize=(20, 20))

        for idx, scores in enumerate(scores_mat):
            scores_np = np.array(scores)
            ax = fig.add_subplot(4, 3, idx+1)
            # append the attention weights
            im = ax.imshow(scores, cmap='viridis')

            fontdict = {'fontsize': 10}

            ax.set_xticks(range(len(self.all_tokens)))
            ax.set_yticks(range(len(self.all_tokens)))

            ax.set_xticklabels(self.all_tokens, fontdict=fontdict, rotation=90)
            ax.set_yticklabels(self.all_tokens, fontdict=fontdict)
            ax.set_xlabel('{} {}'.format(x_label_name, idx+1))

            fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(outfile_path)
        plt.close()

    def explain(self, input, out_file_path, layer):
        input_ids, ref_input_ids = self.construct_input_ref_pair(input)
        token_type_ids, ref_token_type_ids = self.construct_input_ref_token_type_pair(input_ids)
        position_ids, ref_position_ids = self.construct_input_ref_pos_id_pair(input_ids)
        attention_mask = self.construct_attention_mask(input_ids)

        indices = input_ids[0].detach().tolist()
        self.all_tokens = self.__pipeline.tokenizer.convert_ids_to_tokens(indices)
        
        # ground_truth_tokens = self.__pipeline.tokenizer.encode(label, add_special_tokens=False)
        # ground_truth_end_ind = indices.index(ground_truth_tokens[-1])
        # ground_truth_start_ind = ground_truth_end_ind - len(ground_truth_tokens) + 1

        start_scores, output_attentions = self.predict(input_ids,
                                   token_type_ids=token_type_ids, \
                                   position_ids=position_ids, \
                                   attention_mask=attention_mask)

        output_attentions_all = torch.stack(output_attentions)
    
        self.visualize_token2token_scores(output_attentions_all[layer].squeeze().detach().cpu().numpy(), out_file_path)
    
def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint) 
    model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=args.num_labels, output_attentions=True)
    model.eval()
    model.zero_grad()
    device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clf = transformers.pipeline("text-classification", 
                                model=model, 
                                tokenizer=tokenizer, 
                                device=device
                                )
    
    exp_model = ExplainableTransformerPipeline(args.model_checkpoint, clf, device)

    idx=0
    with jsonlines.open(args.a1_analysis_file, 'r') as reader:
        for obj in reader:
            exp_model.explain(obj["review"], os.path.join(args.output_dir,f'example_{idx}'), args.layer)
            idx+=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis_dir', default='out', type=str, help='Directory where attribution figures will be saved')
    parser.add_argument('--model_checkpoint', type=str, default='microsoft/deberta-v3-base', help='model checkpoint')
    parser.add_argument('--a1_analysis_file', type=str, default='out/a1_analysis_data.jsonl', help='path to a1 analysis file')
    parser.add_argument('--num_labels', default=2, type=int, help='Task number of labels')
    parser.add_argument('--output_dir', default='out', type=str, help='Directory where model checkpoints will be saved')    
    parser.add_argument('--layer', default=0, type=int, help='Attribution layer')
    args = parser.parse_args()
    main(args)