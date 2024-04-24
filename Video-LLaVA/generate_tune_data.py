import os
import json
import argparse

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--data_path', help='', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument('--video_folder', help='Name of the folder containing the videos.', required=True)
    
    return parser.parse_args()

def gen_data(args):
    # get data 
    original_data = json.load(open(args.data_path, "r"))
    videos = original_data.items()
    
    # data array
    data = []
    
    # index
    index = 0
    
    # loop through video narration data
    for qid, value in videos:
        # skip redacted
        if (value['status'] == 'redacted'):
            print("Redacted")
            continue
        
        # full summary from first pass
        full_summary1 = ""
        for summary in value["narration_pass_1"]["summaries"]:
            full_summary1 = full_summary1 + summary["summary_text"]
         
        # full summary from second pass   
        full_summary2 = ""
        for summary in value["narration_pass_2"]["summaries"]:
            full_summary2 = full_summary2 + summary["summary_text"]
            
        # choose longest summary
        summary = ""   
        if len(full_summary1) >= len(full_summary2):
            summary = full_summary1
        else:
            summary = full_summary2
            
        # remove artifacts
        summary = summary.replace("#Summary ", "")
        
        # create path
        path = args.video_folder + qid + ".mp4"
        
        # add data if the path exists
        if os.path.exists(path):
          # print that video data is getting converted
          print(f"Converting Video#{index}") 
          print(f"Video QID: {qid}") 
        
          # create video info in format for LLAvA
          data.append(
          {
            "id": f"{index}",
            "video": f"{path}",
            "conversations": [
              {
                "from": "human",
                "value": "What is a summary of the video?\n<video>"
              },
              {
                "from": "ego4D Dataset Narrations",
                "value": f"{summary}"
              }
            ]
          })
          
          # incriment index
          index = index + 1
          continue
        
        print("Video File Missing")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)    
    with open(f"{args.output_dir}/{args.output_name}.json", "w") as json_file:
        json.dump(data, json_file)        

if __name__ == "__main__":
    args = parse_args()
    gen_data(args)