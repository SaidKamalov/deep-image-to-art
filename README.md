# DEEP-IMAGE-TO-ART

A project for Practical Machine Learning and Deep Learning course from Innopolis University.

The main purpose is to examine the existing models for Unsupervised Image-to-Image Translation.

## Project structure

- `api` - folder with API for the project
- `data` - folder with intermediate and temporary data
- `datasets` - folder with datasets for training and testing
- `experiments` - folder with experiments with different models
- `models` - folder with implementations of models
- `src` - folder with source code
- `utils` - folder with helping scripts/functions

## How to run the API

From the project root directory run the following command:

```bash
$ python api/app.py
```

The Flask App will start at [localhost:5000](localhost:5000)

## How to run model inference

From the project root directory run the following command:

```bash
$ python src/inference.py --model_name <model_name_from_available> --input_path <path_to_input_image> --output_path <path_for_output_image>
```

The available models are(case insensitive):

- `CycleGAN` - simple CycleGAN model
- `UGATIT_light` - light version of UGATIT model
- `CUT` - CUT model

## How to create your own dataset

From the project root directory run the following command:

```bash
$ python src/prepare_dataset.py --dataset_name <dataset_name> --path_to_set_A <path_to_set_A> --path_to_set_B <path_to_set_B>
```

The dataset will be save in `datasets` folder by default.

More parameters can be found by calling --help flag.