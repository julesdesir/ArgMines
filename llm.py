# Import modules
import argparse
import os
import re
import PyPDF2
import networkx as nx
import torch
import re
import sys
from tqdm import tqdm
from pdfminer.high_level import extract_pages
import pdfplumber
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
from TF_IDF import tfidf, load_idf
from tesserocr import PyTessBaseAPI
import os
import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import shutil
import spacy
from tqdm import tqdm
from PDF_Convertor import convert_pdf
sys.path.append("/data/thomas/Data Arg/modules")
os.environ['TRANSFORMERS_CACHE'] = '/data/thomas/Data Arg/modules/'

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers



# Build classes
class One_hidden_layer_net(torch.nn.Module):
  def __init__(self, input_size, hidden_neurons, output_size):
      super(One_hidden_layer_net, self).__init__()
      # One hidden layer 
      self.linear_one = torch.nn.Linear(input_size, hidden_neurons)
      self.ReLU = torch.nn.ReLU()
      self.linear_two = torch.nn.Linear(hidden_neurons, output_size) 
      self.softmax = torch.nn.Softmax(dim=1)  # Apply softmax along dimension 1


  def forward(self, x):
      x = self.linear_one(x)
      x = self.ReLU(x)
      logits = self.linear_two(x)
      y_pred = self.softmax(logits)
      return y_pred



class one_layer_net(torch.nn.Module):
    def __init__(self, input_size, hidden_neurons, output_size):
        super(one_layer_net, self).__init__()
        # hidden layer 
        self.linear_one = torch.nn.Linear(input_size, hidden_neurons)
        self.linear_two = torch.nn.Linear(hidden_neurons, output_size) 
        self.softmax = torch.nn.Softmax(dim=1)  # Apply softmax along dimension 1


    def forward(self, x):
        x = self.linear_one(x)
        x = self.linear_two(x)
        y_pred = self.softmax(x)
        return y_pred



class Article_simplifier():
    
    def __init__(self, device = "cuda", model_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'):
        bnb_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side = 'left')
        self.model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, quantization_config = bnb_config, device_map = 'auto', cache_dir = '/data/thomas/Data Arg/modules/')
        self.generate_text = transformers.pipeline(
    model=self.model, tokenizer=self.tokenizer,
    return_full_text=False,  # If using langchain set True
    task="text-generation",
    # Pass model parameters here too
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    top_p=0.15,  # Select from top tokens whose probability add up to 15%
    top_k=0,  # Select from top 0 tokens (because zero, relies on top_p)
    do_sample = True,
    max_new_tokens=512,  # Max number of tokens to generate in the output
    repetition_penalty=1.1  # If output begins repeating increase
)
        self.device = device
    
    
    
    def sentence_extraction(self, text, number):
        response = ''
        i = 0
        with open('extracted/0/section 0', 'w') as f:
            f.write(text)
        for  path in os.listdir(f'extracted/0'):
            ########################################
            # Split the text into sentences
            ########################################
            # Read the text file
            text_paragraph_path = f'extracted/0/' + path # Change the path
            with open(text_paragraph_path, 'r') as f:
                text_paragraph = f.read()

            # Split the text into lines
            lines = text_paragraph.split('\n')

            ########################################
            # First: text embedding and detection of the reasonings (by group of 2 following sentences)
            ########################################

            # Approach: use distilroberta-base fine-tuned on MNLI
            model_name_or_path_MNR_loss = 'finetuning_jules/output/training_nli_v2_distilroberta-base-2024-01-31_15-10-12' # Change the path
            model_MNR_loss = SentenceTransformer(model_name_or_path_MNR_loss)

            sentences = lines           

            embeddings = model_MNR_loss.encode(sentences, convert_to_tensor=True).to('cuda')
            
            print('ok')
            # It returns a NxN matrix with the respective cosine similarity scores for all possible pairs between embeddings1 and embeddings2, with N = len(embeddings1) = len(embeddings2)
            cosine_similarities = util.cos_sim(embeddings, embeddings).to('cuda')
            #Output the pairs with their score

            # List of cosine similarity scores for each couple of sentences
            print(cosine_similarities)
            mean_cosine_score = cosine_similarities.mean()
            std_cosine_score = cosine_similarities.std()
            filtered_idx_cos_sim = []
            cosine_scores = []
            for i in range(len(sentences)):
                for j in range(i+1, len(sentences)):
                    cosine_score = cosine_similarities[i,j]
                    if cosine_score > mean_cosine_score + 2 * std_cosine_score:
                        filtered_idx_cos_sim.append([i,j])
                        cosine_scores.append(cosine_score.to('cpu'))
            filtered_idx_cos_sim = np.array(filtered_idx_cos_sim)
            cosine_scores = np.array(cosine_scores)
            

            groups = []
            deleted_links = []
            for pair in filtered_idx_cos_sim:
                connected_groups = [group for group in groups if any(index in group for index in pair)]

                if not connected_groups:
                    groups.append(set(pair))
                else:
                    new_group = set(pair).union(*connected_groups)
                    groups = [group for group in groups if group not in connected_groups]
                    groups.append(new_group)
            print(groups)
            k = 0
            for group in groups:
                group = list(group)
                if len(group) <= 4:
                    pass
                else:
                    group_embeddings = embeddings.index_select(0, torch.LongTensor(list(group)).to('cuda')).to('cuda')
                    group = torch.tensor(list(group))
                    cosine_similarities = util.cos_sim(group_embeddings, group_embeddings).to('cuda')
                    mean_cosine_score = cosine_similarities.mean()
                    std_cosine_scores = cosine_similarities.std()
                 
                    
                    for i in range(len(group)):
                        for j in range(i+1, len(group)):
                            cosine_score = cosine_similarities[i,j]
                            if cosine_score < mean_cosine_score and ([group[i],group[j]] in filtered_idx_cos_sim or [group[j], group[i]] in filtered_idx_cos_sim):
                                try:
                                    index = np.where(filtered_idx_cos_sim == [min(group[i],group[j]), max(group[i],group[j])])[0][0]
                                    deleted_links.append(filtered_idx_cos_sim[index])

                                    np.delete(cosine_scores, index)
                                    np.delete(filtered_idx_cos_sim, [min(group[i],group[j]), max(group[i],group[j])])
                                except:
                                    pass

            filtered_couples_cos_sim = [sentences[idx[0]]+'. '+sentences[idx[1]] for idx in filtered_idx_cos_sim]
            filtered_sentences1_cos_sim = [sentences[idx[0]] for idx in filtered_idx_cos_sim]
            filtered_sentences2_cos_sim = [sentences[idx[1]] for idx in filtered_idx_cos_sim]
            filtered_embeddings1_cos_sim = embeddings.index_select(0, torch.LongTensor(filtered_idx_cos_sim[:,0]).to('cuda')) # Selection of indices in the first dimension of embeddings1
            filtered_embeddings2_cos_sim = embeddings.index_select(0, torch.LongTensor(filtered_idx_cos_sim[:,1]).to('cuda'))
            with open('cosscores', 'w') as f:
                for i in range(len(cosine_scores)):
                    f.write(f'{filtered_idx_cos_sim[i][0]} - {filtered_idx_cos_sim[i][1]} : {cosine_scores[i]}\n')
            with open('cosscores_weak', 'w') as f:
                for i in range(len(deleted_links)):
                    f.write(f'{deleted_links[i][0]} - {deleted_links[i][1]}\n')
            with open('cosscores_weak', 'r') as f:
                lines = f.readlines()
                unique_lines = set(lines)
            with open('cosscores_weak', 'w') as f:
                for line in unique_lines:
                    f.write(line + '\n')

            with open('cosscores', 'r') as file1:
                lines1 = file1.readlines()

            # Read the content of the second file
            with open('cosscores_weak', 'r') as file2:
                pairs_to_keep = set(file2.readlines())

            # Filter lines from the first file based on the pairs in the second file
            for line in lines1:
                for pairs in pairs_to_keep:
                    if pairs in line:
                        lines1.remove(line)

            # Write the filtered lines to a new file
            with open('cosscores', 'w') as output_file:
                output_file.writelines(lines1)

            ########################################
            # Second: classify the reasonings (entailment / contradiction / entailment)
            ########################################

            # Approach: use softmax classifier trained on the MNLI dataset

            # Define the class for single layer NN
            # Necessary to write it in a cell to make torch.load() work
            embeddings1 = embeddings.to('cuda')
            embeddings2 = embeddings.to('cuda')


            # Load the classifier model trained on MNLI with the server of the campus
            model_classifier_MNLI_path = 'nn_1_hidden_layer_V2_TEST_epoch_100.pt' # Change the path
            model_classifier_MNLI = torch.load(model_classifier_MNLI_path, map_location=torch.device('cuda'))

            distance_dev = torch.abs(filtered_embeddings1_cos_sim - filtered_embeddings2_cos_sim)
            concatenated_dev = torch.cat((filtered_embeddings1_cos_sim, filtered_embeddings2_cos_sim, distance_dev), dim=1)

            with torch.no_grad():
                outputs_classifier = model_classifier_MNLI(concatenated_dev)
                predicted_labels_classifier = torch.argmax(outputs_classifier, dim=1)

            # Comparison with the repartition of entailments / contradictions / neutrals on the whole MNLI dataset
            # to see if the filtering with the cosine similarity is relevant to keep only the reasonings

            distance_dev_WHOLE_ARTICLE = torch.abs(embeddings1 - embeddings2)
            concatenated_dev_WHOLE_ARTICLE = torch.cat((embeddings1, embeddings2, distance_dev_WHOLE_ARTICLE), dim=1)

            with torch.no_grad():
                outputs_classifier_WHOLE_ARTICLE = model_classifier_MNLI(concatenated_dev_WHOLE_ARTICLE)
                predicted_labels_classifier_WHOLE_ARTICLE = torch.argmax(outputs_classifier_WHOLE_ARTICLE, dim=1)

            nb_entailments_WHOLE_ARTICLE = (predicted_labels_classifier_WHOLE_ARTICLE == 0).sum().item()
            nb_contradictions_WHOLE_ARTICLE = (predicted_labels_classifier_WHOLE_ARTICLE == 1).sum().item()
            nb_neutrals_WHOLE_ARTICLE = (predicted_labels_classifier_WHOLE_ARTICLE == 2).sum().item()
            
            print(f'Number of entailments in the whole article: {nb_entailments_WHOLE_ARTICLE} ({nb_entailments_WHOLE_ARTICLE/len(predicted_labels_classifier_WHOLE_ARTICLE)*100}%)')
            print(f'Number of contradictions in the whole article: {nb_contradictions_WHOLE_ARTICLE} ({nb_contradictions_WHOLE_ARTICLE/len(predicted_labels_classifier_WHOLE_ARTICLE)*100}%)')
            print(f'Number of neutrals in the whole article: {nb_neutrals_WHOLE_ARTICLE} ({nb_neutrals_WHOLE_ARTICLE/len(predicted_labels_classifier_WHOLE_ARTICLE)*100}%)')

            ########################################
            # Output: the classified reasonings (entailment / contradiction)
            ########################################
            
            with open(f'linked/0/section 0',  'w') as out_file:
                for idx in range(len(predicted_labels_classifier)):
                    try: 
                        num1 = int(re.search(r'\d+',filtered_sentences1_cos_sim[idx]).group())
                        num2 = int(re.search(r'\d+',filtered_sentences2_cos_sim[idx]).group())
                        if predicted_labels_classifier[idx] == 0:
                            out_file.write(f'{num1}-{num2}: Entailment\n')
                        elif predicted_labels_classifier[idx] == 1:
                            out_file.write(f'{num1}-{num2}: Contradiction\n')

                    except: 
                        pass
            with open(f'extracted/0/' + path, 'w') as out_file:
                
                for sentence in sentences:
                    try:
                        number =  int(re.search(r'\d+',sentence).group())

                        if number in filtered_idx_cos_sim:
                            out_file.write(sentence + '\n')
                    except:
                        pass

            print('output.txt file created.')
             
    
    def contextualisation(self, extracted_text, extracted_lit, extracted_legends):
        sentences = extracted_text.split('\n')
        legends = extracted_lit.split('\n')
        for legend in legends:
            legend_number = int(re.search(r'\d+',legend).group())
            sentences[legend_number] = sentences[legend_number] + '$$$(' + legend.replace(f'{legend_number}', '') + ')'
        pattern = r'\(Fig\. \d+\)'

        res = ''
        for sentence in sentences:
            res += sentence + '\n'
        return res


    def sentence_classification(self,text, number, save = False):
        torch.cuda.empty_cache()
        pattern = r'\n(?=3\.\d+\.)'
        sections = re.split(pattern, text)
        response  = ''
        i = 0
        section_sentences =''
        with open(f'extracted/0/section 0', 'r') as f:
            lines = f.readlines()
            for line in lines:
                section_sentences += line.split('$$$')[0]
        prompt =  f'<s> [INST]You are a helpful assistant that helps classifying sentences extracted from a  text. You must label them as "claim from the litterature", "conclusion made by the authors"or "experiment result". The output has to be formatted like that : "[Sentence number] - [Label]\n[Sentence number] - [label]\n..."  .For example "The high acid ratio is a consequence of the condition in the environment" is a claim from the litterature, "The predicted plankton population is higher in the equatorial regions (Figure 2.b)" is an experimental result, "Sub-tropical waters have ThEi -ratios in the range of ∼0.01 to 0.2, and temperate and sub-polar northern hemisphere waters have ThEi -ratios between ∼0.1 and 0.3." is an experimental result and "This implies that the overall effect of using the f-ratio to calculate global export results in an overestimate of the magnitude of the BCP." is a conclusion made by the authors. [/INST] I have extracted these sentences : {section_sentences}, their classification is [generated output].</s>'
        result = self.generate_text(prompt)
        print(result[0]['generated_text'])
        with open(f'classified/0/section 0', 'w') as f:
            f.write(result[0]['generated_text'])
        i+=1

        if save:
            file = input('Enter the name of the file on which to save the classification results : ')
            with open(file, 'w') as f:
                f.write(response)
        return(response)
    

    def sentence_linking(self, text, number, save = False):
        torch.cuda.empty_cache()
        pattern = r'\n(?=3\.\d+\.)'
        sections = re.split(pattern, text)
        response  = ''
        
        i = 0
        section_sentences =''
        with open(f'extracted/0/section 0', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    section_sentences += line
        with open(f'linked/0/section 0', 'r') as f:
                lines = f.readlines()
                section_links = ''
                for line in lines:
                    section_links += line[:9]

        prompt =  f'<s> [INST]You are a helpful assistant that helps linking sentences extracted from a text. You must label the linked sentences with either "Induction" or "Deduction". Deduction is the application of a rule on an example, induction is the creation of a rule following the observation of a phenomenon. For example, "5 - The high acid ratio is a consequence of the condition in the environment" and "2 - Here the high acid ratio indicates that the environment is polluted" are linked by a deduction thus must be labeled "Deduction", and the output has to feature the line "5 - 2 : Deduction", and "3 - I am observing high sodium ratios in my dead coral samples" and "8 - the sodium ratio indicates that this corals are dead" are linked by an induction thus has to be labeled "Induction" and the output has to feature the line : "3 - 8 : Induction". The output has to be formatted like that : [Number of sentence 2] - [Number of sentence 2] : [label]\n[Number of sentence 1] - [Nuber of sentence 2] : [label]\n ...\n Indeed an output looking like that : "1 - 2 : Deduction\n2 - 3 : Deduction\n1 - 4 : Deduction\n4 - 8 : Deduction\n4 - 9 : Deduction\n7 - 9 : Deduction\n2 - 5 : Deduction\n5 - 6 : Induction\n5 - 7 : Induction\n7 - 8 : Induction" is good whereas an output looking like that : "1 - 2 : Deduction\n2 - 3 : Deduction\n1 - 4 : Deduction\n4 - 8 : Deduction\n4 - 9 : Deduction\n7 - 9 : Deduction\n2 - 5 : Deduction\n5 - 6 : Induction\n5 - 7 : Induction\n7 - 8 : Induction\nExplanation :\n* Sentence 1 describes the results of the study, while sentence 2 provides details about the model predictions based on the results. Therefore, the relationship between these two sentences is one of deduction." is bad, an output looking like that : "1 - 2 : Deduction\n2 - 3 : Deduction\n1 - 4 : Deduction\n4 - 8 : Induction\n1 - 9 : Deduction\n10 - 11 : Deduction\n11 - 12 : Deduction\n12 - 13 : Deduction\n13 - 14 : Deduction\n14 - 15 : Deduction" is bad.[/INST]Now do it for these sentences : "{section_sentences}", with these couples of sentences : {section_links} :  [generated output] </s>' 
        result = self.generate_text(prompt)
        with open(f'linked/0/section 0 bis', 'w') as f:
            f.write(result[0]['generated_text'])
                
            i+=1
        if save:
            file = input('Enter the name of the file on which to save the linking  results: ')
            with open(file, 'w') as f:
                f.write(response)
        return(response)


    def group_linking(self):
        with open('extracted/0/section 0', 'r') as f:
            sentences = f.readlines()
        sentences_and_numbers = {}
        for sentence in sentences:
            try:
                sentence_number = int(re.search(r'\d+',sentence).group())
                try:
                    sentence = sentence.replace(f'{sentence_number}', '').replace('\n', '')
                except Exception as error:
                    print(error)
                try:
                    sentences_and_numbers[sentence] = {}
                    sentences_and_numbers[sentence]['number'] = sentence_number
                except Exception as error:
                    print(error)
            except:
                pass
                
        with open('classified/0/section 0', 'r') as f:
            labels = f.readlines()
        labeled_keys = []
        for label in labels:
            try:
                label_number = int(re.search(r'\d+',label).group())
            except:
                continue   
            for key in sentences_and_numbers.keys():
                if label_number == sentences_and_numbers[key]['number']:
                    sentences_and_numbers[key]['label'] = label
                    labeled_keys.append(key)
        try:
            for key in list(set(sentences_and_numbers.keys()) - set(labeled_keys)):
                sentences_and_numbers[key]['label'] = 'non_labeled'
        except Exception as error:
            print(error)
        linked_idx = []
        with open('cosscores', 'r') as f:
            for line in f.readlines():
                integers = [int(match) for match in re.findall(r'\b\d+\b', line.split(':')[0])]
                linked_idx.append(integers)
        groups = []
        
        for pair in linked_idx:
            connected_groups = [group for group in groups if any(index in group for index in pair)]
            if not connected_groups:
                groups.append(set(pair))
            else:
                new_group = set(pair).union(*connected_groups)
                groups = [group for group in groups if group not in connected_groups]
                groups.append(new_group)
        conclusion_groups = []
        claim_groups = []
        result_groups = []
        for group in groups:
            claim_count = 0
            conclusion_count = 0
            result_count = 0
            for number in group:
                for key in sentences_and_numbers.keys():
                    if number == sentences_and_numbers[key]['number']:
                        label = sentences_and_numbers[key]['label']
                        break
                if re.search(r'\b(Experiment)\b|\b(Experimental Result)\b', label):
                    result_count += 1
                elif re.search(r'\b(Conclusion)\b|\b(prout)\b', label):
                    conclusion_count += 1
                elif re.search(r'\bClaim\b', label):
                    claim_count += 1
            results = np.array([claim_count, conclusion_count, result_count])
            if type(np.argmax(results)) == np.int64:
                if int(np.argmax(results)) == 0:
                    result_groups.append(group)
                elif int(np.argmax(results)) == 1:
                    conclusion_groups.append(group)
                else: 
                    claim_groups.append(group)
        
        print(conclusion_groups, claim_groups, result_groups)         
        with open('conclusion_groups', 'w') as conc, open('claim_groups', 'w') as claim, open('result_groups', 'w') as res:
            for group in conclusion_groups:
                for elt in group:
                    conc.write(f'{elt}\n')
            for group in result_groups:
                for elt in group:
                    res.write(f'{elt}\n')
            for group in claim_groups:
                for elt in group:
                    claim.write(f'{elt}\n')

        idf = load_idf()
        for group in conclusion_groups:
            sentences = []
            for elt in group:
                for key in sentences_and_numbers.keys():
                    if sentences_and_numbers[key]['number'] == elt:
                        sentences.append(key)
                        break
            result_sentences = []
            claim_sentences = []
            for group in result_groups:
                for elt in group:
                    for key in sentences_and_numbers.keys():
                        print(sentences_and_numbers[key],elt)
                        print(sentences_and_numbers[key] == elt)
                        if sentences_and_numbers[key]['number'] == elt:
                            result_sentences.append(key)
            for group in claim_groups:
                for elt in group:
                    for key in sentences_and_numbers.keys():
                        if sentences_and_numbers[key]["number"] == elt:
                            claim_sentences.append(key)
            print(claim_sentences)
            sentences_idf = [tfidf(sentence, idf) for sentence in sentences]
            links = {}
            print(sentences_idf)
            for i in range(len(sentences_idf)):
                claim_score = {}
                result_score = {}
                for claim_sentence in claim_sentences:
                    score = 0
                    for key in sentences_idf[i]:
                        if key in claim_sentence:
                            score+= sentences_idf[i][key]
                    claim_score[claim_sentence] = score
                for result_sentence in result_sentences:
                    score = 0
                    for key in sentences_idf[i]:
                        if key in result_sentence:
                            score+= sentences_idf[i][key]
                    result_score[result_sentence] = score
                mean_claim_score = np.mean([claim_score[key] for key in claim_score.keys()])
                mean_result_score = np.mean([result_score[key] for key in result_score.keys()])
                std_claim_score = np.std(np.array([claim_score[key] for key in claim_score.keys()]))
                std_result_score = np.std(np.array([result_score[key] for key in result_score.keys()]))
                links[sentences[i]] = []
                for key in claim_score.keys():
                    print(claim_score[key], mean_claim_score,std_claim_score)
                    if claim_score[key] > mean_claim_score + 0 * std_claim_score:
                        links[sentences[i]].append(key)
                for key in result_score.keys():
                    if result_score[key] > mean_result_score + 1.5*std_result_score:
                            links[sentences[i]].append(key)

            
        with open('idf_links', 'w') as f:
            for key in links.keys():
                for key2 in sentences_and_numbers.keys():
                    if key == key2:
                        num1 = sentences_and_numbers[key2]['number']
                        break
                for sentence in links[key]:
                    for key2 in sentences_and_numbers.keys():
                        if sentence == key2:
                            num2 = sentences_and_numbers[key2]['number']
                            break
                    f.write(f'{num1} - {num2}\n')



class Graph():
    def __init__(self, number, extraction_file = 'extracted/', classification_file = 'classified/', links_file = 'linked/'):
        self.graph = nx.Graph()
        self.graph.add_node('Claim from the litterature', color = 'red')
        self.graph.add_node('Conclusion of the authors', color='blue')
        self.graph.add_node('Experiment result', color = 'orange')
        self.number = number
        
        for i in range(0, len(os.listdir(extraction_file + '0'))):
            nodes = []
            
            with open(extraction_file + f"{self.number}/section {i}", 'r') as f:
                lines = f.readlines() 
                for line in lines:
                    nodes.append(line.split('$$$')[0])
            with open(classification_file + f"{self.number}/section {i}", 'r') as f: 
                lines = f.readlines() 
                nodes_labels = []
                for line in lines:
                    nodes_labels.append(line)
            with open("cosscores", 'r') as f:
                lines = f.readlines()
                edges2 = []
                for line in lines:
                    integers = [int(match) for match in re.findall(r'\d+', line)]
                    floats = [float(match) for match in re.findall(r'\d+\.\d+', line)]
                    edges2.append((integers, f'{floats}'))
            with open("conclusion_groups", 'r') as f:
                lines = f.readlines()
                conclusion_edge_numbers = []
                for line in lines:
                    try:
                        conclusion_edge_numbers.append(int(line))
                    except:
                        pass
            with open("claim_groups", 'r') as f:
                lines = f.readlines()
                claim_edge_numbers = []
                for line in lines:
                    try:
                        claim_edge_numbers.append(int(line))
                    except:
                        pass
            with open("result_groups", 'r') as f:
                lines = f.readlines()
                result_edge_numbers = []
                for line in lines:
                    try:
                        result_edge_numbers.append(int(line))
                    except:
                        pass

            with open("cosscores_weak", 'r') as f:
                lines = f.readlines()
                edges3 = []
                for line in lines:
                    try:
                        integers = [int(match) for match in re.findall(r'\d+', line)]
                        floats = [float(match) for match in re.findall(r'\d+\.\d+', line)]
                        edges3.append((integers, f'{floats}'))         
                    except:
                        pass
            with open("idf_links", 'r') as f:
                lines = f.readlines()
                edges4 = []
                for line in lines:
                    try:
                        integers = [int(match) for match in re.findall(r'\d+', line)]
                        floats = [float(match) for match in re.findall(r'\d+\.\d+', line)]
                        edges4.append(integers)
                    except:
                        pass

            with open(links_file + f"{self.number}/section {i} bis", 'r') as f:
                lines = f.readlines()
                pattern = r'\d+'
                edges = []

                for line in lines:
                    pattern = r'\d+'
                    numbers = re.findall(pattern, line)
                    numbers = [int(number) for number in numbers]
                    deduction_pattern = r'\b(deduction)\b|\b(Deduction)\b|\b(deduce)\b'
                    induction_pattern = r'\b(induction)\b|\b(Induction)\b|\b(induce)\b'
                    deduction_match = re.search(deduction_pattern, line)
                    induction_match = re.search(induction_pattern, line)
                    if deduction_match:
                        edges.append((numbers, 'pink'))
                    elif induction_match:
                        edges.append((numbers, 'blue'))
            nodes_and_labels = []
            for node, node_label in zip(nodes, nodes_labels):
                claim_pattern = r'\bClaim\b'
                experiment_pattern = r'\b(Experiment)\b|\b(Experimental Result)\b'
                conclusion_pattern = r'\b(Conclusion)\b|\b(prout)\b'
                node_number = int(re.search(r'\d+', node).group())
                color = 'black'
                if node_number in claim_edge_numbers:
                    '''if claim_match:
                        color = 'cadetblue'
                    if deduction_match:
                        color = 'cadetblue1'
                    if experiment_match:
                        color = 'blue'''
                    color = 'red'
                if node_number in conclusion_edge_numbers:
                    '''if claim_match:
                        color = 'chocolate'
                    if deduction_match:
                        color = 'coral'
                    if experiment_match:
                        color = 'crimson'''
                    color = 'blue'

                if node_number in result_edge_numbers:
                    '''if claim_match:
                        color = 'deeppink'
                    if deduction_match:
                        color = 'darkorchid1'
                    if experiment_match:
                        color = 'darkslateblue'''
                    color = 'orange'

                claim_match = re.search(claim_pattern, node_label)
                experiment_match = re.search(experiment_pattern, node_label)
                conclusion_match = re.search(conclusion_pattern, node_label)
                nodes_and_labels.append((node, {"color": color}))
            self.graph.add_nodes_from(nodes_and_labels)
            with open(extraction_file + f"{self.number}/section {i}", 'r') as f:
                line_dict = {}
                lines = f.readlines()
                for line in lines:
                    try:
                        line_number = int(re.search(r'\d+', line).group())
                        line_dict[f'{line_number}'] = line.split('$$$')[0]
                    except:
                        pass
                for edge3, edge2 in zip(edges3, edges2):
                    #if edge[1] == 'pink' : label = "Deductive"
                    #elif edge[1] == 'blue' : label = "Inductive"
                    label2 = edge2[1]
                    try:
                        self.graph.add_edge(line_dict[f'{int(edge2[0][0])}'], line_dict[f'{int(edge2[0][1])}'], color='red', label = label2)

                    except: 
                        print('error')
                        pass
                for edge in edges4:
                    self.graph.add_edge(line_dict[f'{int(edge[0])}'], line_dict[f'{int(edge[1])}'], color='green')
    def plot(self):
        viz = nx.nx_agraph.to_agraph(self.graph)
        viz.draw(f"article graph for PDF {self.number}", prog="dot", format='svg:cairo')



def main(args=None):
    parser = argparse.ArgumentParser(description='Script to sum up scientific article into argumentatives graphs using LLMs')
    parser.add_argument('--task', help='Task to complete, must be either, "convert_pdf",  "total_process", "sentences_extraction " or "graph_drawing"')
    parser.add_argument('--extraction_folder', help='Path to the folder on which the extracted sentences are. Useful only if task is graph_drawing')
    parser.add_argument('--classification_folder', help='Path to the folder on which the classified sentences are. Useful only if task is graph_drawing')
    parser.add_argument('--links_folder', help='Path to the folder on which the links between sentences are. Useful only if task is graph_drawing')
    parser.add_argument('--device', help='The device on which the algorithm will be sent to, default is "cuda"')
    parser.add_argument('--LLM_model', help='The chosen LLM model, default is Mixtral7x8b')
    parser.add_argument('--PDF_list', help='A .txt doc containing the list of all the PDF files on which the algorithm has to work. Useful only if task is "total_process" or "sentences_extraction"')
    parser = parser.parse_args(args)
    
    if parser.task == 'total_process':
        if parser.PDF_list is None:
            print("You have to specify a list of PDF on which the model will work !")
            pass
        elif parser.device is None:
            if parser.LLM_model is None:
                simplifier = Article_simplifier()
            else:
                simplifier = Article_simplifier(model_id=parser.LLM_model)
        else:
            if parser.LLM_model is None:
                simplifier = Article_simplifier(device=parser.device)
            else:
                simplifier = Article_simplifier(device = parser.device, model_id=parser.LLM_model)
        with open(parser.PDF_list, 'r') as f:
            paths = f.readlines()
            try:
                shutil.rmtree('convertor_outputs')
            except:
                pass
            os.mkdir('convertor_outputs')
            for i in range(len(paths)):
                convert_pdf(paths[i].replace('\n', ''), 'DataBase', f'convertor_outputs/text_{i}', f'convertor_outputs/legends_{i}', f'convertor_outputs/litterature_{i}')
                with open(f'convertor_outputs/text_{i}', 'r') as text, open(f'convertor_outputs/legends_{i}', 'r') as legends, open(f'convertor_outputs/litterature_{i}', 'r') as lit:
                    extracted = text.read()
                    extracted_legends = legends.read()
                    lines = lit.readlines()
                    extracted_lit = ''
                    i = 0
                    current_number = int(re.search(r'\d+', lines[0]).group())
                    extracted_lit += f'{current_number} : '
                    while i < len(lines):
                        while int(re.search(r'\d+', lines[i]).group()) == current_number:
                            extracted_lit += lines[i].replace(f'{current_number} : ', '').replace('\n', ' ')
                            i +=1
                        current_number = int(re.search(r'\d+', lines[i]).group())
                        extracted_lit +=f'\n{current_number} : '
                text = simplifier.contextualisation(extracted, extracted_lit, extracted_legends)
                print('Starting extraction')
                simplifier.sentence_extraction(text, i)
                print('Starting classifiacation')
                simplifier.sentence_classification(extracted, i)
                print('Starting Linking')
                simplifier.sentence_linking(text, i)
                simplifier.group_linking()

                graph = Graph(i)
                graph.plot()
    elif parser.task == 'sentences_extraction':
        if parser.PDF_list is None:
            print("You have to specify a list of PDF on which the model will work !")
            pass
        elif parser.device is None:
            if parser.LLM_model is None:
                simplifier = Article_simplifier()
            else:
                simplifier = Article_simplifier(model_id=parser.LLM_model)
        else:
            if parser.LLM_model is None:
                simplifier = Article_simplifier(device=parser.device)
            else:
                simplifier = Article_simplifier(device = parser.device, model_id=parser.LLM_model)
        with open(parser.PDF_list, 'r') as f:
            paths = f.readlines()
            for i in range(len(paths)):
                with open(f'convertor_outputs/text_{i}', 'r') as text, open(f'convertor_outputs/legends_{i}', 'r') as legends, open(f'convertor_outputs/litterature_{i}', 'r') as lit:
                    extracted = text.read()
                    lines = lit.readlines()
                    extracted_lit = ''
                    i = 0
                    current_number = int(re.search(r'\d+', lines[0]).group())
                    extracted_lit += f'{current_number}: '
                    while i < len(lines):    
                        while int(re.search(r'\d+', lines[i]).group()) == current_number:
                            extracted_lit += lines[i].replace(f'{current_number} : ', '').replace('\n', ' ')
                            i +=1
                            print(i, len(lines))
                            if i == len(lines) - 1:
                                break
                        if i == len(lines) - 1: 
                            break
                        current_number = int(re.search(r'\d+', lines[i]).group())
                        extracted_lit +=f'\n{current_number} : '

                    extracted_legends = legends.read()
                text = simplifier.contextualisation(extracted, extracted_lit, extracted_legends)
                print('Starting extraction')
                simplifier.sentence_extraction(text, i)
                print('Starting classifiacation')
                #simplifier.sentence_classification(extracted, i) 
                print('Starting Linking')
                #simplifier.sentence_linking(text, i)
                simplifier.group_linking()
    elif parser.task == 'graph_drawing':
        if (parser.extraction_folder is None) or (parser.classification_folder is None) or (parser.links_folder is None):
            print('One of the folder name has not been specified, trying with default setting')
            for i in range(len(os.listdir('classified/'))):
                graph = Graph(number=i)
                graph.plot()
        else:
            for i in range(len(os.listdir('parser.extraction_folder'))):
                graph = Graph(number = i, extraction_file = parser.extraction_folder, classification_file = parser.classification_folder, links_file = parser.links_folder)
                graph.plot()
    elif parser.task == 'convert_pdf':
        if parser.PDF_list == None:
            print('You have to specify a list of PDF to convert !')
        else:
            try:
                shutil.rmtree('convertor_outputs')
            except: 
                pass
            os.mkdir('convertor_outputs')
            with open(parser.PDF_list, 'r') as f:
                links = f.readlines()
                for i in range(len(links)):
                    convertor = PdfConvertor(links[i].replace('\n', ''))
                    convertor.forward(f'convertor_outputs/text_{i}', f'convertor_outputs/legends_{i}') 



if __name__ == '__main__':
    main()