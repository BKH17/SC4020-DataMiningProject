# Training Script for BGE and PCA Visualization

This repository contains the scripts to fine-tune a BGE (BAAI General Embedding) model and visualize the resulting text embeddings using Principal Component Analysis (PCA).

## Training of BGE

This section outlines the steps required to set up the environment and run the training script to fine-tune the BGE model.

### 1. Environment Setup

Before running the scripts, please set up the necessary Python environment.

* **Activate your Conda environment:**
    It is recommended to use a Conda environment to manage dependencies. Replace `<your_env_name>` with the name of your environment.

    ```bash
    conda activate <your_env_name>
    ```

* **Install required packages:**
    Install all the required Python libraries using the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

### 2. Run the Training Script

Once the environment is set up, you can start the model fine-tuning process by running the main training script.

```bash
python finetune_BGE.py
```

This will start the training process based on the configuration specified within the script. Make sure your training data is correctly placed as expected by the script.

---

## Visualization of PCA

After fine-tuning your model, you can visualize how the model organizes embeddings in a 2D space. This process involves generating example data, creating embeddings for it, and then plotting the results using PCA.

Follow these steps in order:

### 1. Generate Examples

First, run the script to generate the example query and documents that will be used for visualization.

```bash
python generate_examples.py
```

This will create the necessary text files in the designated output folder bundle.json.

### 2. Generate Embeddings

Next, use the fine-tuned model to generate dense vector embeddings for the examples created in the previous step.

```bash
python inference.py
```

This script will load your model, process the text files, and save the resulting embeddings.

### 3. Generate 2D PCA Plot

Finally, run the visualization script to apply PCA to the embeddings and generate a 2D scatter plot.

```bash
python pca.py
```

This will produce a file showing the 2D PCA projection of your query and document embeddings.