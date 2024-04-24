import json
import re
'''
def correct_answers(answer):
	removals = []
	periods = [x.start() for x in re.finditer('.', answer)]
	for x in periods:
		if x < len(answer) - 1:
			if answer[x+1] == '.':
				removals 
'''
def main():
	with open('ego4d_tune_data.json', 'r') as file:
		annotations = json.load(file)
	ego = []
	for entry in annotations:
		ann = {
			'id': entry['id'],
			'video_id': entry['video'],
			'q': entry['conversations'][0]['value'],
			'a':entry['conversations'][1]['value'],
		}
		ego.append(ann)

	with open('ego4d_gpt_data.json', 'w') as file:
		json.dump(ego, file)

if __name__=="__main__":
	main()
'''
[{"id": "0", 
"video": "/lustre/fs1/home/cap6412.student13/ego4d/ego4d/v2/full_scale/0b637191-00b7-4fd8-af37-d26d80dc59ce.mp4", 
"conversations": [{"from": "human", 
		"value": "What is a summary of the video?\n<video>"}, 
		{"from": "ego4D Dataset Narrations", 
		"value": "C was at the supermarket she walked and checked on prices of the item prices..C was in a supermarket, walked around in the supermarket, viewed and adjusted items on the supermarket shelving.C was in the supermarket, walked around the supermarket and conversed with man QC was inside a room where she operated a phone, talked to person A and looked at the fridgeC sat, ate & interacted with cashier X in the shop"}]}
'''
