# Based on example Eval script from Video-LLaVA repo
# Original Repo: https://github.com/PKU-YuanGroup/Video-LLaVA
import math
import os
import argparse
import json

import torch
import transformers
from tqdm import tqdm
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN
from videollava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from videollava.model.builder import load_pretrained_model
from videollava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from videollava.train.train import smart_tokenizer_and_embedding_resize


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--cache_dir', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)

    return parser.parse_args()

def get_model_output(model, video_processor, tokenizer, video, qs, args):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_VID_START_TOKEN + ''.join([DEFAULT_IMAGE_TOKEN]*8) + DEFAULT_VID_END_TOKEN + '\n' + qs
    else:
        qs = ''.join([DEFAULT_IMAGE_TOKEN]*8) + '\n' + qs

    conv_mode = "llava_v1"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    # print(video_tensor.shape)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)


    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor],
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    #print(outputs)
    return outputs


def run_inference(args):
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    model = model.to(args.device)

    gt_questions = json.load(open(args.gt_file_question, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    gt_answers = json.load(open(args.gt_file_answers, "r"))
    # gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)

    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results

    index = 0
    correct = 0
    answer_counts = [0, 0, 0, 0, 0]
    choice_counts = [0, 0, 0, 0, 0]
    
    # Iterate over each sample in the ground truth file
    for q_uid, answer in gt_answers.items():
        video_name = q_uid
        
        # get data for sample
        qdata = next(item for item in gt_questions if item["q_uid"] == q_uid)
        question = qdata['question']
        options = [qdata['option 0'], qdata['option 1'], qdata['option 2'], qdata['option 3'], qdata['option 4']]
        
        # print data
        print(f"Benchmark Video #{index}")
        print(f"Video ID: {video_name}")
        print(f"Correct Awnser: {answer}")
        
        id = q_uid
        index += 1  
        
        # full question V1
        #question = f"""Which of the following 0 to 4 options best awnsers the following question. Specify the best option number only.
        #Question: {question} 
        #0: {options[0]} 
        #1: {options[1]} 
        #2: {options[2]}
        #3: {options[3]}
        #4: {options[4]}"""

        # full question V2
        question = f"""Which of the following 0 to 4 options best awnsers the following question. Consider each option carefully, since choosing the right option is important. Once a decision is made, specify the best option number only.
        Question: {question} 
        0: {options[0]} 
        1: {options[1]} 
        2: {options[2]}
        3: {options[3]}
        4: {options[4]}"""

        # full question V3
        #question = f"""Consider the following question in relation to the video. What option from 0 to 4 best answers the question? Consider each option carefully, since choosing the right option is vital. Make sure to take note of the differences between the various options since they may be similarily worded, but have varying levels of accuracy to the video. Specify the best option number only.
        #Question: {question} 
        #0: {options[0]} 
        #1: {options[1]} 
        #2: {options[2]}
        #3: {options[3]}
        #4: {options[4]}"""

        # full question V4
        #question = f"""Which of the following 0 to 4 options best awnsers the following question? Consider each option carefully, since choosing the right option is important. If there is no clear best answer, choose one of the options at random. Once a decision is made, specify the best option number only.
        #Question: {question} 
        #0: {options[0]} 
        #1: {options[1]} 
        #2: {options[2]}
        #3: {options[3]}
        #4: {options[4]}"""

        # full question V5
        #question = f"""Consider the following question. Question: {question}
        #Which of the following 0 to 4 options best answers the question? 0: {options[0]}  1: {options[1]}  2: {options[2]} 3: {options[3]} 4: {options[4]}
        #Consider each option carefully, since choosing the right option is important. Once a decision is made, specify the option number only.
        #"""

        # full question V6
        #question = f"""Consider the following question. Question: {question}
        #Which of the following 0 to 4 options best answers the question? 0: {options[0]}  1: {options[1]}  2: {options[2]} 3: {options[3]} 4: {options[4]}
        #Consider each option carefully, since choosing the right option is important. If not sure, or there is no clear best option randomly choose between 0 to 4. Once a decision is made, specify the option number only.
        #"""

        sample_set = {'id': id, 'question': question, 'answer': answer}

        # Load the video file
        #for fmt in tqdm(video_formats):  # Added this line
        temp_path = os.path.join(args.video_dir, f"{video_name}.mp4")#{fmt}
        if os.path.exists(temp_path):
            video_path = temp_path
            # try:
            # Run inference on the video and add the output to the list          
            output = get_model_output(model, processor['video'], tokenizer, video_path, question, args)
            
            # print sample eval data
            print(f"Predicted Awnser: {output}")
            
            sample_set['pred'] = output
            output_list.append(sample_set)
            ans_file.write(json.dumps(sample_set) + "\n")
            
            if str(answer) in output:
                correct = correct + 1

            # count awnser distribution
            if '0' in str(answer):
                answer_counts[0] += 1
          
            elif '1' in str(answer):
                answer_counts[1] += 1
            
            elif '2' in str(answer):
                answer_counts[2] += 1
            
            elif '3' in str(answer):
                answer_counts[3] += 1
            
            elif '4' in str(answer):
                answer_counts[4] += 1
             
            # count choice distribution   
            if '0' in output:
                choice_counts[0] += 1
          
            elif '1' in output:
                choice_counts[1] += 1
            
            elif '2' in output:
                choice_counts[2] += 1
            
            elif '3' in output:
                choice_counts[3] += 1
            
            elif '4' in output:
                choice_counts[4] += 1
            
    print(f"Accuracy: {correct/index}")
    print("")
    print(f"Option 0 Actual: {answer_counts[0]}")
    print(f"Option 1 Actual: {answer_counts[1]}")
    print(f"Option 2 Actual: {answer_counts[2]}")
    print(f"Option 3 Actual: {answer_counts[3]}")
    print(f"Option 4 Actual: {answer_counts[4]}")
    print("")
    print(f"Option 0 Chosen: {choice_counts[0]}")
    print(f"Option 1 Chosen: {choice_counts[1]}")
    print(f"Option 2 Chosen: {choice_counts[2]}")
    print(f"Option 3 Chosen: {choice_counts[3]}")
    print(f"Option 4 Chosen: {choice_counts[4]}")
    
    ans_file.close()
    #Save the output list to a JSON file
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
        json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
