## Fraud's Bargain Attack (FBA)
Code for our paper "Fraud's Bargain Attack to Textual Classifier via Metropolis-Hasting Sampler".

### Requirements
This experiments are done based on the Huggingface (https://huggingface.co/) and pytorch. To set up the propoer environment, you may run the 'requirements.txt' with the following command:
```
pip install requirements.txt 
```
### Usage
* There are three datasets: AG's News, Emotion, and SST2 in the [dataset] folder. If you wanna use your own data, please save them with torch .pt file.
* To perform the attack, you can run the following command:
```
python main.py --num_sample num_samples\
 --save_dir save_dir \
 --dataset dataset_path \
 --victim_model_name huggingface_model_name \
 --num_class num_class\
 --Lambda lambda
```
