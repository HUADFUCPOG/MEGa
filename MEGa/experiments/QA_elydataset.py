# had to comment out from transformers.processing_utils import Unpack in src/modeling_llama_gated.py since transformers2 python interpreter gave some dependcy issues
from transformers import AutoTokenizer
from datasets import Dataset
import torch
from collections import defaultdict
import gc
import numpy as np
from tqdm import tqdm
import sys
from torch.utils.data import Dataset, DataLoader
sys.path.append('../../src')
from modeling_llama_gated import LlamaGatedForCausalLM
sys.path.append('../../dataset')
from ely_stories import passage_list, sampled_first_sentences, sampled_post_first_sentence_passages, paraphrased_list, memory_queries, list_of_qa_pairs_per_story, claude_reading_comp_q, theme_list
# from wiki_stories import passage_list, sampled_first_sentences, sampled_post_first_sentence_passages, paraphrased_list, claude_reading_comp_q, theme_list


import os
from openai import OpenAI
import time
## Setting Random Seed for Reproducability, Jan 17th ##
import torch
import numpy as np
import random
import pandas as pd
import pickle
import torch.nn.functional as F  # Ensure you have this import at the top
import requests
import json
import os
from openai import OpenAI 
from numpy.linalg import norm
from gatedlora_model import GatedLoraModel
from peft import LoraConfig
import httpx
import time
from datasets import load_dataset


SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
## Setting Random Seed for Reproducability, Jan 17th ##
print(len(passage_list))
print(len(paraphrased_list))
print(len(claude_reading_comp_q)) # should be 3*50 = 150

current_time = time.strftime("%Y%m%d%H%M%S")

target_modules = [f"model.layers.{l}.mlp.up_proj" for l in list(range(0, 32))] + [f"model.layers.{l}.mlp.down_proj" for l in list(range(0, 32))]
config = LoraConfig(
    r=128, # was prev 200
    lora_alpha=128, # was prev 200
    target_modules=target_modules,
    lora_dropout=0.05, # was prev 0.05
    bias="none",
)
print(f"LoRA Rank: {config.r}")


#####################################################   
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
learning_rate = 3e-5
num_train_epochs = 10
save_path = "./results.pkl"
#####################################################

tokenizer = AutoTokenizer.from_pretrained(model_name)
llama_gated = LlamaGatedForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager") # use multiple GPUs, as per Xu's slack comment
# llama_gated.to("cuda")
# might need the above line, commented it out to debug
# turn llama model into gated lora model
lora_llama_gated = GatedLoraModel(llama_gated, config, adapter_name="lora_0")

print('New Cell\n\n\n')


for name, param in lora_llama_gated.injected_modules[0].named_parameters():
    print(name, param.shape)

print('New Cell\n\n\n')

# specify which layer to use for embedding
lora_llama_gated.set_embedding_module(lora_llama_gated.injected_modules[-2]) # last up_proj

##########################################################################################
########################### helper functions #############################################
##########################################################################################
# Test assuming not knowing which gated lora to use. Need to calculate the lora weights based on the 
# using top k
def generate_general(correct_idx, prompt, k, actual_answer, retrieval_extra_prompt='Answer should be short and concise.'):
    # gen_kwargs = {"max_length": 200, "do_sample": False, "pad_token_id": tokenizer.eos_token_id} # was prev 200
    gen_kwargs = {"max_new_tokens": 200, "do_sample": False, "pad_token_id": tokenizer.eos_token_id} # was prev 200


    # formated_query = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|>"
    formated_query = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{prompt}" # This is changed.
    tokenized_query = tokenizer(formated_query, return_tensors="pt").to("cuda")
    query_embedding, model_output = lora_llama_gated.get_embedding(tokenized_query["input_ids"])
    query_KV_cache = model_output.past_key_values

    max_sims = []
    max_indices = []
    for stacked_tensors in lora_llama_gated.lora_context_key.values():
        stacked_tensor = torch.stack(stacked_tensors)  # Shape: (num_gpt_gen_queries_per_story, feature_dim)
        # print(f"size of stacked_tensor is {stacked_tensor.size()}") # this is (num_gpt_gen_queries + 1, 4096)
        dot_products = torch.matmul(stacked_tensor, query_embedding)

        max_val, max_idx = torch.max(dot_products, dim=0)
        max_sims.append(max_val)
        max_indices.append(max_idx.item())


    lora_weights = torch.nn.functional.softmax(torch.tensor(max_sims), dim=0) # convert max_sims from list to tensor

    print(lora_weights)


    correct_gate = 0

    if lora_weights.argmax() == correct_idx:
        correct_gate = 1

   
    formated_query_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{prompt}{retrieval_extra_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    tokenized_query_prompt = tokenizer(formated_query_prompt, return_tensors="pt").to("cuda")
    # get KV cache of the query
    with torch.no_grad():
        # outputs = lora_llama_gated.generate(**gen_kwargs, **tokenized_query_prompt, gate_mode=-2, lora_weights=lora_weights)
        outputs = lora_llama_gated.generate(**gen_kwargs, **tokenized_query_prompt, gate_mode=-2, lora_weights=lora_weights, past_key_values=query_KV_cache) 



    generated_answer = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
    # Extract the generated answer text
    start_token = "<|start_header_id|>assistant<|end_header_id|>"
    start_index = generated_answer.find(start_token) + len(start_token)
    cleaned_generated_answer = generated_answer[start_index:].strip("<|eot_id|>")
    

    # Calculate log probability of actual_answer
    full_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{prompt}{retrieval_extra_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{actual_answer}"
    tokenized_full_prompt = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    full_ids = tokenized_full_prompt["input_ids"] # shape [1, seq_len]
    
    with torch.no_grad():
        outputs_full = lora_llama_gated(input_ids=full_ids, gate_mode=-2, lora_weights=lora_weights, past_key_values=query_KV_cache)
        logits = outputs_full.logits
        # log_probs = torch.log_softmax(logits, dim=-1)
    

    # 2) We only want the part of the logits corresponding to the answer tokens
    #    i.e. exclude the user/prompt tokens at the front.
    actual_answer_ids = tokenizer(actual_answer, return_tensors="pt").input_ids.to("cuda")
    answer_len = actual_answer_ids.shape[1]       # number of tokens in the answer
    total_len = full_ids.shape[1]                # total number of tokens in prompt+answer
    answer_start = total_len - answer_len        # where the answer starts

    # 3) "Teacher-forcing" shift: we gather from logits up to the last token
    #    but only in the answer region. We also shift the labels by 1.
    #    So if the answer tokens are at indices [answer_start ... total_len-1],
    #    the relevant logits are [answer_start ... total_len-2].
    shift_logits = logits[:, answer_start : total_len - 1, :]          # shape [1, answer_len-1, vocab_size]
    shift_labels = full_ids[:, answer_start + 1 : total_len].contiguous()  # shape [1, answer_len-1]

    # 4) Log-probs and gather
    shift_log_probs = torch.log_softmax(shift_logits, dim=-1)
    log_prob = shift_log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # [1, answer_len-1]

    log_prob_avg = log_prob.mean().item()
    
    return correct_gate, lora_weights, cleaned_generated_answer, log_prob_avg



#### General Capabilities Test #### 
# Test assuming not knowing which gated lora to use. Need to calculate the lora weights based on the 
# Note, need to store first sentences -- is this in line with Haim's section 2 "Synthesizing new memories into knowledge" as a RAG analog
# arguably, better since don't need to store entire story, just the first sentence


print(f"Finished exact context retrieval; now moving onto questions that may not exactly align with t_1, ..., t_c\n\n")


#9.5: compute cosine_sim and auto grading


# obv never used in production
api_key = os.getenv("OPENAI_API_KEY")

# Taken from the first comment of ps://medium.com/@akhilkanugolu/dealing-with-certificate-issues-in-the-openai-package-on-your-local-mhttachine-a7f563394c6c

client = OpenAI(api_key=api_key,http_client=httpx.Client(verify=False))

def cosine_sim(story1, story2):
    while True:
        try:
            response1 = client.embeddings.create(
                input= story1,
                model="text-embedding-3-large"
            )

            v1 = np.array(response1.data[0].embedding)

            response2 = client.embeddings.create(
                input= story2,
                model="text-embedding-3-large"
            )

            v2 = np.array(response2.data[0].embedding)

            cosine_sim_res = np.dot(v1,v2)/(norm(v1)*norm(v2))
            return cosine_sim_res
        except:
            print("API Call err, sleeping!!!!!!!!!!!")
            time.sleep(5)



# def generate_queries_for_story(question, model_ans, actual_ans):
#     # GPT as a judge, binary output



def generate_queries_for_story(story, question, model_ans, actual_ans):
    # GPT as a judge, binary output
    prompt = f"You are evaluating a prospective answer to a question on a given article. The answer should be relatively short and to the point. The answer should NOT be the whole article or a rewrite of the article. Your grading is binary: give 0 if the prospective answer is incorrect, give 1 if the prospective answer is correct. Your output is either 0 or 1, no other information should be in the output.\n\nThe article: {story}\n\nThe question: {question}\n\nThe correct answer: {actual_ans}\n\nThe prospective answer is {model_ans}"

    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Call error: {e}, retrying in 5 seconds...")
            time.sleep(5)

   

## finish 9.5: cosine_sim and autograding

# for q in claude_reading_comp_q:
#     generate_general(q, k = 42069)

############ For MMLU ############
def format_prompt(example, subject):
            # Format the prompt following Llama-3 style
    question = example['question']
    choices = [example['choices'][i] for i in range(4)]
    formatted_query = f"""<|start_header_id|>system<|end_header_id|>
The following are multiple choice questions (with answers) about {subject}. Only respond with the letter of the correct answer.
<|start_header_id|>user<|end_header_id|>Question: {question}

Choose the correct answer from the following options:
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}"""
    formatted_query_answer = f"""<|start_header_id|>system<|end_header_id|>
The following are multiple choice questions (with answers) about {subject}. Only respond with the letter of the correct answer.
<|start_header_id|>user<|end_header_id|>Question: {question}

Choose the correct answer from the following options:
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The answer is"""
    return formatted_query, formatted_query_answer

def evaluate_example(example, subject, model):
    ############### Get lora weights from question embedding
    # formated_query = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{example['question']}<|eot_id|>"
    formated_query, formated_query_answer = format_prompt(example, subject)
    
    tokenized_query = tokenizer(formated_query, return_tensors="pt").to("cuda")
    tokenized_query_answer = tokenizer(formated_query_answer, return_tensors="pt").to("cuda")
    
    query_embedding, model_output = lora_llama_gated.get_embedding(tokenized_query["input_ids"])
    query_KV_cache = model_output.past_key_values
    
    max_sims = []
    max_indices = []
    
    for stacked_tensors in lora_llama_gated.lora_context_key.values():
        stacked_tensor = torch.stack(stacked_tensors)  # Shape: (num_gpt_gen_queries_per_story, feature_dim)
        # print(f"size of stacked_tensor is {stacked_tensor.size()}") # this is (num_gpt_gen_queries + 1, 4096)
        # print(f"stacked_tensor is {stacked_tensor}")
        # print(f"query_embedding is {query_embedding}")
        dot_products = torch.matmul(stacked_tensor, query_embedding)

        max_val, max_idx = torch.max(dot_products, dim=0)
        max_sims.append(max_val)
        max_indices.append(max_idx.item())

        # max_sims.append(torch.max(dot_products)) # using the max for now, change later to top-k if neccesary
    # print(f"max_indices are {max_indices}")
    lora_weights = torch.nn.functional.softmax(torch.tensor(max_sims), dim=0) # convert max_sims from list to tensor

    # weights2 = torch.zeros_like(lora_weights)
    # weights2[lora_weights.argmax()] = 1

    # lora_weights = weights2

    # print(f"lora weights are {lora_weights}")
    
    ############### Get answer from model
    gen_kwargs = {"max_new_tokens": 10, "temperature": 0.0, "do_sample": False, "pad_token_id": tokenizer.eos_token_id} # was prev 200
    with torch.no_grad():
        outputs = lora_llama_gated.generate(
            **gen_kwargs,
            **tokenized_query_answer,
            gate_mode=-2,
            lora_weights=lora_weights,
            past_key_values=query_KV_cache
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    predicted_answer = generated_text.split("The answer is")[-1][:3]
    # print(predicted_answer)

    # Extract the letter answer (A, B, C, or D)
    for letter in ['A', 'B', 'C', 'D']:
        if letter in predicted_answer[:2]:  # Check first two characters
            predicted_idx = ord(letter) - ord('A')
            break
    else:
        return False

    correct_idx = example['answer']
    return predicted_idx == correct_idx

def mmlu_eval_one_step(model):
    subjects = ["abstract_algebra", "anatomy", "astronomy", "business_ethics"]  # Add more subjects as needed
    results = {}

    for subject in subjects:
        dataset = load_dataset("cais/mmlu", subject, split="test")
        # Evaluate the model on the dataset
        correct_count = 0
        total_count = len(dataset)

        for example in dataset:
            if evaluate_example(example, subject, model):
                correct_count += 1

        # Calculate accuracy for this subject
        accuracy = correct_count / total_count
        results[subject] = accuracy
        # print(f"{subject} Accuracy: {accuracy:.2%}")

    # Print overall results
    # print("\nOverall Results:")
    # for subject, accuracy in results.items():
    #     print(f"{subject}: {accuracy:.2%}")
    # print(f"Average Accuracy: {sum(results.values()) / len(results):.2%}")
    
    return results



def evaluation(total_stories = 50):

    ###########################################################
    ###################### QA EVALUATION ######################
    ###########################################################
    QA_results = []
    # 10. Finally, loop over each question, run the QA chain, and print the answers
    for idx, (question, actual_answer) in enumerate(claude_reading_comp_q[:total_stories*3], start=0): # skip the grading on the reconstructions, first q should be 'What event brought...'
        correct_gate, lora_weights, answer, avg_log_prob = generate_general(idx // 3, question, k = 1, actual_answer=actual_answer) # this generate_general is generate_general_one_pass!!!


        perplexity = torch.exp(-torch.tensor(avg_log_prob))
        # print(f"Q{idx}: {question}\nA: {answer}\n---\n")
        # answers_list.append(answer)
        cos_sim = cosine_sim(answer, actual_answer)
        # cos_sim_list.append(cos_sim)
        # print(f"{idx} Cos Sim: {cos_sim}")
        gpt_judge_res = generate_queries_for_story(passage_list[idx//3], question, answer, actual_answer) # get the index of the relevant story, we use //3 since there are 3 questions per story, 50 stories!
        # binary_auto_grades.append(gpt_judge_res)
        # print(f"GPT-Judge: {gpt_judge_res}")

        # print(f"Q{idx}: {question}\nModel-A: {answer}\nActual-A: {actual_answer}\nCorrect Gate: {correct_gate}\nAvg Log Prob: {avg_log_prob:.4f}\nPerplexity: {perplexity:.4f}\nCosine Sim: {cos_sim:.4f}\nGPT-Judge: {gpt_judge_res}\n---\n")
        QA_results.append({
            "question": question,
            "model_answer": answer,
            "actual_answer": actual_answer,
            "correct_gate": correct_gate,
            "avg_log_prob": avg_log_prob,
            "perplexity": perplexity,
            "cosine_sim": cos_sim,
            "gpt_judge_res": gpt_judge_res
        })
    

    print("Moving towards story reconstructions from first sentence")
    ###########################################################
    ##### Reconstruction EVALUATION from first sentence #######
    ###########################################################


    reconstruction_first_sentence_results = []
    for idx, first_sen in enumerate(sampled_first_sentences[:total_stories], start=0): 
        question = first_sen
        actual_answer = passage_list[idx]
        correct_gate, lora_weights, answer, avg_log_prob = generate_general(idx, question, k = 1, retrieval_extra_prompt = "\nReconstruct the entire story:", actual_answer=actual_answer) # this generate_general is generate_general_one_pass!!!

        perplexity = torch.exp(-torch.tensor(avg_log_prob))
        cos_sim = cosine_sim(answer, actual_answer)
        # cos_sim_list.append(cos_sim)
        # print(f"{idx} Cos Sim: {cos_sim}")
        gpt_judge_res = generate_queries_for_story(passage_list[idx], question, answer, actual_answer) # get the index of the relevant story, we use //3 since there are 3 questions per story, 50 stories!
        # binary_auto_grades.append(gpt_judge_res)
        # print(f"GPT-Judge: {gpt_judge_res}")

        # print(f"Q{idx}: {question}\nModel-A: {answer}\nActual-A: {actual_answer}\nCorrect Gate: {correct_gate}\nAvg Log Prob: {avg_log_prob:.4f}\nPerplexity: {perplexity:.4f}\nCosine Sim: {cos_sim:.4f}\nGPT-Judge: {gpt_judge_res}\n---\n")
        reconstruction_first_sentence_results.append({
            "question": question,
            "model_answer": answer,
            "actual_answer": actual_answer,
            "correct_gate": correct_gate,
            "avg_log_prob": avg_log_prob,
            "perplexity": perplexity,
            "cosine_sim": cos_sim,
            "gpt_judge_res": gpt_judge_res
        })
    

    print("Moving towards story reconstructions from QUESTION, as discussed w Haim on Jan28")
    reconstruction_fromquestion_results = [] # as discussed with Haim on Jan 28
    # the following indexing is to do 50 Q&A total (ie first question of each story, as each story has 3 questions
    # and testing 150 questions is very-time consuming. we get the point with 50 stories, and mark total_stories as we go)
    for idx, (question, actual_answer) in enumerate([claude_reading_comp_q[:total_stories*3][i] for i in range(0, len(claude_reading_comp_q[:total_stories*3]), 3)] , start=0):
        question = question
        actual_answer = passage_list[idx] # the target is the original story
        correct_gate, lora_weights, answer, avg_log_prob = generate_general(idx, question, k = 1, actual_answer=actual_answer, retrieval_extra_prompt = "\nReconstruct the entire story that is related to the above question.")

        perplexity = torch.exp(-torch.tensor(avg_log_prob))

        cos_sim = cosine_sim(answer, actual_answer)
        gpt_judge_res = generate_queries_for_story(passage_list[idx], question, answer, actual_answer) # get the index of the relevant story, we use //3 since there are 3 questions per story, 50 stories!
       
        reconstruction_fromquestion_results.append({
            "question": question,
            "model_answer": answer,
            "actual_answer": actual_answer,
            "correct_gate": correct_gate,
            "avg_log_prob": avg_log_prob,
            "perplexity": perplexity,
            "cosine_sim": cos_sim,
            "gpt_judge_res": gpt_judge_res
        })
    

   
    ###########################################################
    ######################## MMLU EVAL #######################
    ###########################################################
    mmlu_results = mmlu_eval_one_step(lora_llama_gated)
    
    QA_results_df = pd.DataFrame(QA_results)
    reconstruction_first_sentence_results_df = pd.DataFrame(reconstruction_first_sentence_results)
    reconstruction_fromquestion_results_df = pd.DataFrame(reconstruction_fromquestion_results)
    # reconstruction_theme_results_df = pd.DataFrame(reconstruction_theme_results)
    
    return QA_results_df, reconstruction_first_sentence_results_df, reconstruction_fromquestion_results_df, mmlu_results


##########################################################################################
########################### Training loop ###############################################
##########################################################################################

results = []
# for story_index, paraphrased_list_that_story, (story, qa_pairs) in enumerate(zip(passage_list, paraphrased_list, list_of_qa_pairs_per_story)):
for story_index, (story, paraphrased_list_that_story) in enumerate(zip(passage_list, paraphrased_list)):
    # batch_size = len(qa_pairs) // 4  # should be 20 questions and answers per story
    batch_size = 2 # was prev 1
    # print(f"batch_size is {batch_size}")
    print(f"========== Training on story {story_index} ==========")
    if story_index > 0:
        # adding a new adapter, 0 already has a lora adapter
        lora_llama_gated.peft_config.update({f"lora_{story_index}": config}) # run this before inject_adapter
        lora_llama_gated.inject_adapter(lora_llama_gated, f"lora_{story_index}")
    
    class StoryDataset(Dataset):
        def __init__(self, paraphrased_list):
            self.samples = []
            for paraphrase in paraphrased_list:
                # Build conversation text
                text = (
                    f"<|begin_of_text|>"
                    f"<|start_header_id|>user<|end_header_id|>"
                    f"Reconstruct the story:" # was prev was Reconstruct the story
                    f"<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>"
                    f"{paraphrase}"
                    f"<|eot_id|>"
                )
                
                tokenized = tokenizer(text, return_tensors="pt").to("cuda")
                input_ids = tokenized["input_ids"][0]
                attention_mask = tokenized["attention_mask"][0]

                # Mask out user tokens in labels
                labels = input_ids.clone()
                
                # Find the assistant portion index
                assistant_start = text.find("<|start_header_id|>assistant<|end_header_id|>")
                user_part = tokenizer(text[:assistant_start], return_tensors="pt")
                user_len = user_part["input_ids"].shape[1]

                # Set user tokens to -100
                labels[:user_len] = -100

                self.samples.append({
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask
                })

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    # dataset = QADataset(qa_pairs)
    story_raw_text_dataset = StoryDataset(paraphrased_list_that_story)
    def collate_fn(batch):
        """
        Collate a list of samples into a single batch.
        We need to pad input_ids and labels to the same length.
        """
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            # print('tis None')
            pad_token_id = tokenizer.eos_token_id
        # else:
        #     print('tis NOT None')

        input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]

        # Pad them so they match the longest sequence in the batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, 
            batch_first=True, 
            padding_value=-100
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, 
            batch_first=True, 
            padding_value=0
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }


    data_loader_stories = DataLoader(
        story_raw_text_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )


    question_re_story_list = [passage_list[story_index]]

    trainable_params = [p for p in lora_llama_gated.parameters() if p.requires_grad]
    # print(f"num of trainable_params is {len(trainable_params)}")
  
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    formatted_questions_list = [tokenizer(f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{i}<|eot_id|>", return_tensors="pt").to("cuda") for i in question_re_story_list]
   
    context_key = [j["input_ids"] for j in formatted_questions_list]
    # context_key.append(tokenized_query["input_ids"]) # don't forget the first sentence also being part of the context key, in addition to reading comp questions
   
    # get and set context key
    lora_llama_gated.set_lora_context_key(f"lora_{story_index}", context_key) # context_key is a list of tensors of different sizes, but will crystalize to the same size after getting the embedding

    # training loop
    for epoch in range(num_train_epochs):
        
        for batch_i, batch in enumerate(data_loader_stories):
            # Move batch to GPU
            input_ids = batch["input_ids"].to("cuda") # was prev cuda:0
            labels = batch["labels"].to("cuda") # was prev cuda:1
            attention_mask = batch["attention_mask"].to("cuda") # was prev cuda:0

            outputs = lora_llama_gated(input_ids=input_ids, labels = labels, use_cache=True, attention_mask=attention_mask, gate_mode=story_index, lora_weights=None, pad_token_id=None)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(f"(Paraphrases) Story {story_index} | Epoch {epoch} | Batch {batch_i} | Loss {loss.item():.4f}")
    
    (QA_results, reconstruction_first_sentence_results, reconstruction_fromquestion_results, mmlu_results) = evaluation(total_stories = story_index+1)
    results.append((QA_results, reconstruction_first_sentence_results, reconstruction_fromquestion_results, mmlu_results))
    # with open(f'/n/holylabs/LABS/sompolinsky_lab/Everyone/xupan/ely/gated-LoRA copy/test/Ely_dataset_results_ihatetime_{current_time}.pkl', 'wb') as file:
    with open(save_path, 'wb') as file:
        pickle.dump(results, file)





