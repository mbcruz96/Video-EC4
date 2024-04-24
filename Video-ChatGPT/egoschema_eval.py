import os
import argparse
import json
from tqdm import tqdm
from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.inference import video_chatgpt_infer

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--projection_path", type=str, required=True)

    return parser.parse_args()


def run_inference(args):
    """
    Run inference EgoSchema on DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name,
                                                                                        args.projection_path)
    # Load both ground truth file containing questions and answers
    with open(args.gt_file_question) as file:
        gt_questions = json.load(file)
    with open(args.gt_file_answers) as file:
        gt_answers = json.load(file)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results
    conv_mode = args.conv_mode
    index = 0
    correct = 0
    answer_counts = [0, 0, 0, 0, 0]
    choice_counts = [0, 0, 0, 0, 0]
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    index = 0
    for sample in tqdm(gt_questions):
        video_name = sample['q_uid']
        options = [sample['option 0'], sample['option 1'], sample['option 2'], sample['option 3'], sample['option 4']]
        question = sample['question']
        id = sample['google_drive_id']

        if video_name not in gt_answers.keys():
                continue
        answer = gt_answers[video_name]
      
        index += 1
	
        # print data
        print(f"Benchmark Video #{index}")
        print(f"Video ID: {video_name}")
        print(f"Correct Awnser: {answer}")
	
        # full question V1
        #question = f""""Select one of the following options that best answers the following question. Specify a single number 0 to 4 coorelating to the best option as a response.
        #Question: {question}
        #0: {options[0]}
        #1: {options[1]}
        #2: {options[2]}
        #3: {options[3]}
        #4: {options[4]}"""
        
        # full question V2
        #question = f"""Which of the following 0 to 4 options best answers the following question. Specify the best option number only.
        #Question: {question}
        #0: {options[0]}
        #1: {options[1]}
        #2: {options[2]}
        #3: {options[3]}
        #4: {options[4]}"""

        # full question V3
        #question = f"""Which of the following 0 to 4 options best answers the following question. Consider each option carefully, since choosing the right option is important. Once a decision is made, specify the best option number only.
        #Question: {question}
        #0: {options[0]}
        #1: {options[1]}
        #2: {options[2]}
        #3: {options[3]}
        #4: {options[4]}"""

        # full question V4
        #question = f"""Consider the following question in relation to the video. What option from 0 to 4 best answers the question? Consider each option carefully, since choosing the right option is vital. Make sure to take note of the differences between the various options since they may be similarily worded, but have varying levels of accuracy to the video. Specify the best option number only.
        #Question: {question}
        #0: {options[0]}
        #1: {options[1]}
        #2: {options[2]}
        #3: {options[3]}
        #4: {options[4]}"""

        # full question V5
        #question = f"""Which of the following 0 to 4 options best answers the following question? Consider each option carefully, since choosing the right option is important. If there is no clear best answer, choose one of the options at random. Once a decision is made, specify the best option number only.
        #Question: {question}
        #0: {options[0]}
        #1: {options[1]}
        #2: {options[2]}
        #3: {options[3]}
        #4: {options[4]}"""

        # full question V6
        #question = f"""Consider the following question in relation to the video. What option from 0 to 4 best answers the question? Consider each option carefully, since choosing the right option is vital. Make sure to take note of the differences between the various options since they may be similarily worded, but have varying levels of accuracy to the video. Specify the best option number only.
        #Question: {question}
        #0: {options[0]}
        #1: {options[1]}
        #2: {options[2]}
        #3: {options[3]}
        #4: {options[4]}"""

        # full question V7
        #question = f"""Which of the following 0 to 4 options best answers the following question? Consider each option carefully, since choosing the right option is important. If there is no clear best answer, choose one of the options at random. Once a decision is made, specify the best option number only.
        #Question: {question}
        #0: {options[0]}
        #1: {options[1]}
        #2: {options[2]}
        #3: {options[3]}
        #4: {options[4]}"""

        # full question V8
        #question = f"""Consider the following question. Question: {question}
        #Which of the following 0 to 4 options best answers the question? 0: {options[0]}  1: {options[1]}  2: {options[2]} 3: {options[3]} 4: {options[4]}
        #Consider each option carefully, since choosing the right option is important. Once a decision is made, specify the option number only.
        #"""

        # full question V9
        question = f"""Consider the following question. Question: {question}
        Which of the following 0 to 4 options best answers the question? 0: {options[0]}  1: {options[1]}  2: {options[2]} 3: {options[3]} 4: {options[4]}
        Consider each option carefully, since choosing the right option is important. If not sure, or there is no clear best option randomly choose between 0 to 4. Once a decision is made, specify the option number only.
        """

        sample_set = {'id': id, 'question': question, 'answer': answer}

        # Load the video file
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        # Check if the video exists
        if os.path.exists(video_path):
            video_frames = load_video(video_path)

        try:
            # Run inference on the video and add the output to the list
            output = video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower,
                                             tokenizer, image_processor, video_token_len)
            sample_set['pred'] = output
            output_list.append(sample_set)

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

        except Exception as e:
            print(f"Error processing video file '{video_name}': {e}")

    print(f"Number of samples: {index}")
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

    # Save the output list to a JSON file
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
        json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
