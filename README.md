## Fraud's Bargain Attack (FBA)
Code for our paper "Fraud's Bargain Attack to Textual Classifier via Metropolis-Hasting Sampler".

### Requirements
This experiments are done based on the Huggingface (https://huggingface.co/) and pytorch. To set up the propoer environment, you may run the `requirements.txt` with the following command:
```
pip install requirements.txt 
```
### Usage
* In the experiments, we tend to attack 3 public datasets: AG's News, Emotion, and SST2. These three datasets are available on the [Google Drive](https://drive.google.com/drive/folders/1D6qd93IuPt7IUFszkZ5vEgyja0jL869o?usp=sharing), and one can download and place them in the `\datasets`. If you wanna use your own data, please save them with torch .pt file.
* First run the following codes, to generate the word embeddings:
```
python bert_emd_create.py
```
* To perform the attack, you can run the following command:
```
python main.py --num_sample num_samples\
 --save_dir save_dir \
 --dataset dataset_path \
 --victim_model_name huggingface_model_name \
 --num_class num_class\
 --Lambda lambda
```
