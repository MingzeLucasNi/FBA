# Fraud's Bargain Attack: Generating Adversarial Text Samples via Word Manipulation Process
This repository contains Pytorch implementations of the IEEE Transactions on Knowledge and Data Engineering (TKDE) 2024 paper: Fraud's Bargain Attack: Generating Adversarial Text Samples via Word Manipulation Process. [paper link](https://www.computer.org/csdl/journal/tk/5555/01/10384773/1TzvOmedR60)
## Abstract
Recent research has revealed that natural language processing (NLP) models are vulnerable to adversarial examples. However, the current techniques for generating such examples rely on deterministic heuristic rules, which fail to produce optimal adversarial examples. In response, this study proposes a new method called the Fraud's Bargain Attack (FBA), which uses a randomization mechanism to expand the search space and produce high-quality adversarial examples with a higher probability of success. FBA uses the Metropolis-Hasting sampler, a type of Markov Chain Monte Carlo sampler, to improve the selection of adversarial examples from all candidates generated by a customized stochastic process called the Word Manipulation Process (WMP). The WMP method modifies individual words in a contextually-aware manner through insertion, removal, or substitution. Through extensive experiments, this study demonstrates that FBA outperforms other methods in terms of attack success rate, imperceptibility and sentence quality.

## Requirements
These experiments are done based on the Huggingface (https://huggingface.co/) and pytorch. To set up the proper environment, you may run the `requirements.txt` with the following command:
```
pip install requirements.txt 
```
## Usage
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
## Citation

When using this code, or the ideas of FBA, please cite the following paper
<pre><code>@article{ni2024fraud,
  title={Fraud's Bargain Attack: Generating Adversarial Text Samples via Word Manipulation Process},
  author={Ni, Mingze and Sun, Zhensu and Liu, Wei},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2024},
  publisher={IEEE}
}
</code></pre>


## Contact

Please contact Mingze Ni at firstname.lastname@uts.edu.au or [Wei Liu](https://wei-research.github.io/) at firstname.lastname@uts.edu.au if you're interested in collaborating on this research!
