import json 
import re

def main():
	# opening evaluation predictions
	with open('video_chatgpt_activitynet_qa_preds.json', 'r') as file:
		answers = json.load(file)
	
	results = []	# stores all predications that selected a multiple choice option
	samples = 0	# total number of samples in eval
	
	# file loop to get predictions with number answers
	for id, entry in enumerate(answers):
		# finding option number in predictions
		temp = re.findall(r'\d+', entry['pred'])
		if len(temp) > 0:
			entry["pred_answer"] = temp[0]
			results.append(entry)
			samples += 1
	
	# saving evalutions to results.json
	with open('results_V9.json', 'w') as file:
		json.dump(results, file)

	correct = 0
	answer_counts = [0, 0, 0, 0, 0]
	choice_counts = [0, 0, 0, 0, 0]
	# looping through evaluations with number predictions to calculate accuracy
	for entry in results:
		# count correct answers
		answer = str(entry['answer'])
		output = str(entry['pred_answer'])

		if answer == output:
			correct += 1
		
		# count awnser distribution
		if '0' == answer:
			answer_counts[0] += 1

		elif '1' == str(answer):
			answer_counts[1] += 1

		elif '2' == str(answer):
			answer_counts[2] += 1
	
		elif '3' == str(answer):
			answer_counts[3] += 1

		elif '4' == str(answer):
			answer_counts[4] += 1

		# count choice distribution
		if '0' == output:
			choice_counts[0] += 1

		elif '1' == output:
			choice_counts[1] += 1

		elif '2' == output:
			choice_counts[2] += 1

		elif '3' == output:
			choice_counts[3] += 1

		elif '4' == output:
			choice_counts[4] += 1
		
	# printing metrics to results.txt
	print(f"Number of samples: {samples}")
	print(f"Accuracy: {correct/samples}")
	print("")
	print("Ground truth distribution")
	print(f"Option 0 Actual: {answer_counts[0]}")
	print(f"Option 1 Actual: {answer_counts[1]}")
	print(f"Option 2 Actual: {answer_counts[2]}")
	print(f"Option 3 Actual: {answer_counts[3]}")
	print(f"Option 4 Actual: {answer_counts[4]}")
	print("")
	print("Predicted Distribution")
	print(f"Option 0 Chosen: {choice_counts[0]}")
	print(f"Option 1 Chosen: {choice_counts[1]}")
	print(f"Option 2 Chosen: {choice_counts[2]}")
	print(f"Option 3 Chosen: {choice_counts[3]}")
	print(f"Option 4 Chosen: {choice_counts[4]}")

if __name__=="__main__":
	main()
