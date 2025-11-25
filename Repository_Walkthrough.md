# Install Dependencies

1. Create virtual environment
2. `pip install -r requirements.txt`
3. Git clone FastChat and install its dependencies
```
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e ".[llm_judge]"
```

# Obtaining ShareGPT dataset
4. Obtain ShareGPT dataset
You need to have git lfs installed to obtain the full dataset
Run `git lfs install`
Then `git clone https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered`
Due to the potential difficulty of installing git lfs onto greatlakes
To put this dataset onto Greatlakes a workaround would be to download it on your computer and then use Globus to move the dataset to Greatlakes
5. If you are Greatlakes run `module load cuda`

Now you will be ready to executed the commands listed in command-list.md

A suggestion for model answer generation would be to have a copy of the llm_judge directory for each model you are testing.
