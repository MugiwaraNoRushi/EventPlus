# EventPlus: A Temporal Event Understanding Pipeline

Please refer to our preprint for details. [[PDF]](https://arxiv.org/pdf/2101.04922.pdf)

```
@article{ma2021eventplus,
  title = {EventPlus: A Temporal Event Understanding Pipeline},
  author = {Ma, Mingyu Derek and Sun, Jiao and Yang, Mu and Huang, Kung-Hsiang and Wen, Nuan and Singh, Shikhar and Han, Rujun and Peng, Nanyun},
  journal = {arXiv preprint arXiv:2101.04922},
  year = {2021}
}
```

## Quick Start with ISI shared NAS

If you are using the system on a machine with access to ISI shared NAS, you could directly activate environment and copy the code and start using it right away!

```
# 1 - Environment Installation: Activate existing environment
conda activate /nas/home/mingyuma/miniconda3/envs/event-pipeline-dev

# 2 - Prepare Components (Submodules): Copy the whole codebase
cp -R /nas/home/mingyuma/event-pipeline/event-pipeline-dev YOUR_PREFERRED_PATH

# 2.5 - In background: Run REST API for event duration detection module for faster processing
(optional) tmux new -s duration_rest_api
conda activate /nas/home/mingyuma/miniconda3/envs/event-pipeline-dev
cd component/REST_service
python main.py
(optional) exit tmux window

# 3 - Application 1: Raw Text Annotation.
#     The input is a multiple line raw text file, 
#     the output pickle and json file will be saved to designated paths
cd YOUR_PREFERRED_PATH/project
python APIs/test_on_raw_text.py -data YOUR_RAW_TEXT_FILE -save_path SAVE_PICKLE_PATH -save_path_json SAVE_JSON_PATH -negation_detection

# 4 - Application 2: Web App for Interaction and Visualization
#     A web app will be started and user can input a piece of text
#     and get annotation result and visualization
cd YOUR_PREFERRED_PATH/project
tmux new -s serve
python manage.py runserver 8080
```

## Quick Start with Independent Environment and Dependencies

You can also choose to load each component from their corresponding GitHub repo, but some of them may need permission to clone. The dependent submodules are listed [here](https://github.com/PlusLabNLP/event-pipeline/blob/master/.gitmodules).

0 - Clone the codebase with all submodules

```
git clone --recurse-submodules https://github.com/PlusLabNLP/event-pipeline.git
# or use following commands
git clone https://github.com/PlusLabNLP/event-pipeline.git
git submodule init
git submodule update
```

1 - Environment Installation

Change prefix (last line) of `env.yml` to fit your path, then run

```
conda env create -f env.yml
conda activate event-pipeline
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_jnlpba_md-0.2.4.tar.gz
python -m spacy download en_core_web_sm
pip install git+https://github.com/hltcoe/PredPatt.git
```

2 - Prepare Components (Submodules) 

For `component/BETTER` module, download the trained model [[Link]](https://drive.google.com/file/d/19_W6azeG5KRQxLDICswqwIFX0QOjxh_L/view?usp=sharing), unzip and place it under `component/BETTER/joint/worked_model_ace`. Also:

```
# before cloning the repo
export GIT_LFS_SKIP_SMUDGE=1
```

For `component/Duration` module, download `scripts` zip file [[Link]](https://drive.google.com/file/d/1s1uLcQjjFdfcto3BZ3aRi8pPzLf9KELe/view?usp=sharing), unzip and place it under `component/Duration/scripts`.

3 & 4 - The instructions for step 3 and 4 are the same as using the ISI shared NAS.

## Deployment as Web Service

Here are instruction of how to deploy the web application on an server

### Set up web server

```
pip install uwsgi
```

If you met errors like `error while loading shared libraries libssl.so.1.1`, reference [this link](https://www.bswen.com/2018/11/others-Openssl-version-cause-error-when-loading-shared-libraries-libssl.so.1.1.html) and do the following

```
export LD_LIBRARY_PATH=/nas/home/mingyuma/miniconda3/envs/event-pipeline/lib:$LD_LIBRARY_PATH
```

### Server port setting

External port: 443 (for HTTPS)

Django will forward traffic from 443 port to internal 8080 port

Internal port
* 8080: run Django main process
* 17000: run service for duration (if we run a REST API for duration module, but now the newer version doesn't need such a separate service)

## Others

Save Environment

```
conda env export > env.yml
conda env update --file env.yml
conda install --yes --file conda_env.txt
python -m spacy download en_core_web_sm
```
