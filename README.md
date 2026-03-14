# AI-Augmented Confocal Laser Endomicroscopy for Rapid Intraoperative Diagnosis of Brain Tumors: A Prospective Multicenter Evaluation
Official implementation of "AI-augmented CLE Imaging" in npj Digital Medicine.

## Environment Setup

You can easily set up the required environment using the provided `environment.yml` file with conda:

```bash
# Create the environment from the environment.yml file
conda env create -f environment.yml

# Activate the newly created environment
conda activate cle_imaging
```

## Preparation

Before running the code, please download the trained model checkpoints from the following link and place them in the `trained_models/` directory:
[Download Trained Models](https://drive.google.com/drive/folders/16iVVFM3njSklbzbihqdGV17m4CvUi4ue?usp=sharing)

## Usage

You can use the `visualization_instance.py` script to run inference on a single image and generate a GradCAM visualization.

### Command Line Arguments

* `--image_path`: (Required) The path to the single image file you want to process.
* `--save_dir`: (Optional) The directory where the resulting visualization will be saved. Defaults to `./visualization_results/`.
* `--annotation`: (Optional) Annotation type for pattern generation. Can be "N", "C", or left empty for default pattern.
* `--model_paths`: (Optional) A list of paths to the trained model checkpoints. The first model in the list is used as the main model for GradCAM generation. Defaults to a list of models in `trained_models/`.

### Example Command

```bash
python visualization_instance.py --image_path "path/to/your/image.jpg"
```

