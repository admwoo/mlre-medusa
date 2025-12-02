# Environment Setup

All tests were done on JupyterLab in the Great lakes environment with the python 3.11-anaconda/2024.02 distribution with 1 gpu of the spgpu partition and 48 GB of RAM.

Start by git cloning this repo into that environment.

# Install Dependencies

1. Create virtual environment using python 3.11 `python -m venv venv`
2. Pip install the requirements file `pip install -r requirements.txt`
3. Git clone FastChat and install its dependencies
```
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e ".[llm_judge]"
```

Then `pip install anthropic`

# Obtaining ShareGPT dataset
4. Obtain ShareGPT dataset

You need to have git lfs installed to obtain the full dataset

Run `git lfs install`
Then `git clone https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered`

Due to the potential difficulty of installing git lfs onto greatlakes,
to put this dataset onto Greatlakes a workaround would be to download the dataset on your computer and then use Globus to move the dataset to Greatlakes
6. 

If you are in Greatlakes then run `module load cuda`

Now you will be ready to execute the commands listed in command-list.md

A suggestion for model answer generation would be to have a copy of the llm_judge directory for each model you are testing.

# Miscenallous Sections
- Unsuccessful Medusa-Hydra Hydra code can be found in medusa-hydra-hybrid-proposed-extension
- An example of valid generated model answers can be found for the MLP extension in llm_judge_example_MLP_Extension
