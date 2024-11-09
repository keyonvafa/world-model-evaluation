# Evaluating World Models in Transformers
Source code for the paper ["Evaluating the World Model Implicit in a Generative Model"](https://arxiv.org/abs/2406.03689) by Keyon Vafa, Justin Y. Chen, Jon Kleinberg, Sendhil Mullainathan, Ashesh Rambachan

### Download data and model checkpoints
Download the data from [this link](https://drive.google.com/drive/folders/1crGsllw1Ha_6dYkswSQW9kddxmel0D4a?usp=share_link). Place the three folders (`shortest-paths`, `noisy-shortest-paths`, and `random-walks`) in a directory called `data`.

Download model checkpoints from [this link](https://drive.google.com/drive/folders/14Vn1jwi5tZ3K6193-brCZnRBp6SUAcWu?usp=share_link). Place the three folders (`shortest-paths`, `noisy-shortest-paths`, and `random-walks`) in a directory called `ckpts`. Alternatively, to train the models yourself, you can use the script `train.py`.

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

The code to reproduce the maps of the world model reconstructed from sequences output by a learned model are contained in the `mapping` folder. The sequences produced by a model trained on shortest paths, noisy shortest paths, and random walks are in the `sequences` folder. These will be the sequences used for map reconstruction.

Run `make_graphs.py` to produce the recontructed graph of Manhattan roads. The graph will be saved as a `.pkl` file in the `graphs/` folder. The script accepts the following flags:
* `--dataset`: the name of the sequence dataset in `mapping/sequences/` to run on
* `--degree`: the maximum degree of an intersection in the reconstructed graph
* `--distance`: the maximum distance (in miles) of a new edge in the reconstructed graph
* `--nsequences`: the number of sequences to use in the recontruction algorithm
* `--randomerr`: if this parameter is in [0,1], the reconstruction algorithm only runs on sequences corresponnding to the true world model but chooses a random fraction (given by this parameter) to corrupt (by default this parameter is turned off)

To produce the visualization of the graphs as maps of Manhattan, run the `make_maps.py` script which will create a `.html` interactive map in the `maps/` folder for each graph in the `graphs/` folder.

We also provide code in the base folder to reproduce sequences rather than using the ones already provided in `mapping/sequences/`. To do so, run
```
python generate_sequences_for_map
```
with the `--data` flag set to `shortest-paths`, `noisy-shortest-paths`, or `random-walks`. Put the resulting sequences into the `mapping/sequences/` folder and run the code described above from `make_graphs.py` to produce graphs and maps from these sequences.

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
