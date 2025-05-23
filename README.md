# Bridging the Gap: Adapting Video Language Models for Egocentric Understanding
**Project Goal:** To enhance the capability of state-of-the-art Large Video Language Models (LVLMs) in comprehending egocentric (first-person perspective) video content, addressing the limitations of models primarily trained on third-person perspectives. 

**Problem Statement:**
Traditional LVLMs, trained on third-person video datasets, struggle with the unique challenges of egocentric videos. These challenges include rapid camera movement, high scene complexity, and the critical need to infer context from subtle visual cues, often involving hands and objects.  This gap limits applications in fields like assistive technology, augmented reality (AR), and human-computer interaction (HCI). 

**Approach & Methodology:**
Our solution, Video-EC4, focused on benchmarking and adapting existing LVLMs, specifically Video-LLaVA and Video-ChatGPT, for egocentric video understanding. 

**Benchmarking:** Initially evaluated Video-LLaVA and Video-ChatGPT on the EgoSchema dataset, which features multiple-choice long-context Video QA, to assess their zero-shot capabilities. 
**Fine-tuning:** Fine-tuned Video-LLaVA (chosen due to Video-ChatGPT's zero-shot performance deficiencies) on a subset of the Ego4D dataset.  Ego4D is a large-scale egocentric video dataset providing diverse scenes, people, and activities. 
**Custom Annotation:** Developed a custom annotation script to convert Ego4D narrations into a two-actor conversational format suitable for fine-tuning, combining full video segment narrations into a single summary.  This approach aimed to provide a rich feature set for the model. 
**Prompt Engineering:** Explored and refined prompts to guide model predictions, specifically to mitigate a bias towards choosing option 0 in Video-LLaVA during benchmarking, leading to accuracy gains. 

**Key Contributions & Results:**
Demonstrated the significant potential of fine-tuning LVLMs on egocentric datasets like Ego4D to enhance their ability to comprehend first-person video content. 

Benchmarking results for Video-LLaVA and Video-ChatGPT showed comparable performance to other models in their tier, highlighting their baseline capabilities. 
Successfully utilized prompt tuning to improve the accuracy and reduce prediction bias in Video-LLaVA's zero-shot multiple-choice Video QA performance. 

**Challenges & Limitations:**
**Counting and Enumeration:** LVLMs exhibited typical difficulties with counting and generating enumerated guesses. 
**Zero-Shot Performance:** Models performed poorly on zero-shot multiple-choice long-context Video QA tasks, with Video-LLaVA often defaulting to the first option when uncertain, and Video-ChatGPT failing to select an answer consistently. 
**Video Segmentation for Fine-tuning:** The fine-tuning pipeline was impacted by the models' method of dividing video samples into equal partitions, which did not adequately capture major events occurring at various moments. 
**Annotation Utilization:** The reliance on voice annotations from Ego4D, broken into key events, led to a loss of clip-specific information when appended into a single video annotation for training. 

**Future Directions:**
**Dynamic Video Processing:** Implement dynamic video processing for Video-LLaVA, selecting frames that align with sub-steps of narration to capture richer contextual information from video segments. 
**Task-Specific Adaptation:** Further explore the models' potential on specific tasks within egocentric data, such as hand recognition, leveraging Ego4D's detailed annotations. 
