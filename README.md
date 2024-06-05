# Evaluating World Models in Transformers
Source code for the paper "Evaluating the World Model Implicit in a Generative Model" by Keyon Vafa, Justin Chen, Jon Kleinberg, Sendhil Mullainathan, Ashesh Rambachan

### Download data and model checkpoints
Download the data from [this link](https://drive.google.com/drive/folders/1crGsllw1Ha_6dYkswSQW9kddxmel0D4a?usp=share_link). Place the three folders (`shortest-paths`, `noisy-shortest-paths`, and `random-walks`) in a directory called `data`.

Download model checkpoints from [this link](https://drive.google.com/drive/folders/14Vn1jwi5tZ3K6193-brCZnRBp6SUAcWu?usp=share_link). Place the three folders (`shortest-paths`, `noisy-shortest-paths`, and `random-walks`) in a directory called `ckpts`.

### Installing dependencies
To install the dependencies, run the following command:
```
pip install -r requirements.txt
```

### Traversal capabilities

To assess traversal capabilities (i.e. which percent of sequences from the model are valid and which percent of valid sequences are shortest paths), run the following command:
```
python evaluate_traversal_capabilities.py
```
You can pass in a `--data` flag with the values `shortest-paths`, `noisy-shortest-paths`, or `random-walks` to evaluate the model on different datasets. You can also pass in a `--use-untrained-model` flag to evaluate the untrained model.



### Existing metrics: next-token test and probe

The next token test assesses the percent of model predictions that are valid turns when prompted with a partial sequence. To run the next token test, run the following command:
```
python next_token_test.py
```

The probe test trains a probe on the model's representation to predict state (i.e. the current intersection implied by the sequence). To run the probe test, run the following command:
```
python probe_test.py
```

For both tests, you can pass in a `--data` flag with the values `shortest-paths`, `noisy-shortest-paths`, or `random-walks` to evaluate the model on different datasets. You can also pass in a `--use-untrained-model` flag to evaluate the untrained model.

### Proposed evaluation metrics: compression and distinction
To evaluate the compression metric on the navigation dataset, run the following command:
```
python compression_test.py
```
The script accepts the following flags:  
* `--data`: the dataset to evaluate on, among `shortest-paths`, `noisy-shortest-paths`, or `random-walks` (default is `shortest-paths`)
* `--use-untrained-model`: whether to evaluate the untrained model (default is to evaluate the trained model)
* `--num-suffix-samples`: the number of suffixes to sample from the model
* `--epsilon`: the threshold for acceptance (default is 0.01)
* `--num-trials`: the number of trials to run (default is 100)

To run the distinction metric on the navigation dataset, run the following command:
```
python distinction_test.py
```
The script accepts the following flags:
* `--data`: the dataset to evaluate on, among `shortest-paths`, `noisy-shortest-paths`, or `random-walks` (default is `shortest-paths`)
* `--use-untrained-model`: whether to evaluate the untrained model (default is to evaluate the trained model)
* `--max-suffix-length`: the maximum suffix length to consider when approximating the true Myhill-Nerode boundary (default is 5)
* `--num-suffix-samples`: the number of suffixes to sample from the model to approximate its Myhill-Nerode boundary (default is 5)
* `--epsilon`: the threshold for acceptance (default is 0.01)
* `--num-trials`: the number of trials to run (default is 100)

### Mapping sequences
To reproduce our maps of Manhattan, first generate samples from the model by running
```
python generate_sequences_for_map
```
with the `--data` flag set to `shortest-paths`, `noisy-shortest-paths`, or `random-walks`. 



### Detour analysis
To analyze detours, run the following command:
```
python detour_analysis.py
```
The script accepts the following flags:
* `--data`: the dataset to evaluate on, among `shortest-paths`, `noisy-shortest-paths`, or `random-walks` (default is `shortest-paths`)
* `--detour-prob`: the probability of taking a detour (default is 0.01)
* `--detour-type`: the type of detour to take, among `random_valid`, `least_likely`, and `second_most_likely` (default is `random_valid`)

### Proposed evaluation metrics on Othello
Make sure the synthetic dataset and model checkpoints are installed using the [instructions here](https://github.com/likenneth/othello_world). We evaluate distinction and compression using the following script:
```
python othello_world/test_world_model.py
```
The `--model` flag can be set to `untrained`, `championship`, or `synthetic`.

We thank Kenneth Li and the authors of the [Othello world model paper](https://arxiv.org/abs/2210.13382) for [making their code and data available](https://github.com/likenneth/othello_world).

### LLM test
The script `logic_puzzles.py` contains the code to perform our evaluation metrics on the seating logic puzzle described in the paper. Before running the code, make sure the following lines are uncommented and set to your API keys:

```
together_api_key = ...
open_ai_api_key = ...
```

To change model names, change the value of the `model_name` variable. The script will output the results of the capabilities evaluation along with the compression and precision tests.

