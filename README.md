# Data free model integration

[![contributions welcome](https://img.shields.io/badge/contributions-welcome-blue?style=plastic)]()
![GitHub repo size](https://img.shields.io/github/repo-size/Guy-Shapira/DeepInversion?style=plastic)
### Report:
Full project report can be found [here](report.pdf)

## Environment setup:
From the root folder of the project run
```
conda env create -f environment.yml
conda activate deep-inversion
```

## Implementations:
* Implemnentation of the creation process DeepInverted images from multiple pretrained models is provided in src/run_get_images.py. An example for tranformation from images to PyTorch's tensors can be seen in src/utils/image_to_batch.py
* Our Implemnentation for creation of multiple CIFAR100 teachers (split the learning task between them) is provided in src/partial_cifar_train_teacher.py. Using src/run_create_teachers.py the creation process can be automated.
* Model integration from multiple teachers is given in src/combined_model.py, the path from which teachers checkpoints are taken should be given in checkpoint_dir.

## Usage:
### Step 1: Training the teachers models

If you already have pretrained models skip to Step 2.
The default in this work is training  ResNet architecture as teachers’ model over distinct parts of CIFAR100.
If you wish to use another architecture, you can change the used model and data loader in the **partial_cifar_train_teacher.py**.
If you wish to reproduce this work results, run the script **python run_create_teacher.py  --n={} --resnet={}**
Where *n* is the number of parts to split CIFAR100 and *resnet* should be either of {18,34,50} and will be the teachers architecture.
Running this script will create n teachers and save them to the path:
*./resnet{args.resnet}split_cifar_to{args.n}/start:{start_class}_end:{end_class}*.


### Step 2: Generating “Deep Inverted” images

In order to generate augmented images from the pretrained teachers run:
**python run_get_images.py  --bs={} --bn={} --teacher_weights={} --prefix={} --max_label={}**
Where *bs* is the batch size of images to create, *bn* is the number of batches of images to generate, *teacher_weights* is a path from which teachers checkpoints will be loaded  *prefix* is the directory path to save the generated images, *max_label* is the number of classes to create images from. The rest of the arguments can be changed according to the explanation that can be found in [here](https://github.com/NVlabs/DeepInversion).
The images would be saved to *./{args.prefix}/best_images*.

### Step 3: Combining the teachers to a student

In order to combine the pretrained teachers (of Step 1) to a student using the generated images (of Step 2), run the following script:
**python combined_model.py --bs={} - -num_batches={} --n= --num_classes={} --num_epochs={} --teachers_arch={} --checkpoint_dir={} --data_dir={} --save_path={}**
Where *bs* is the batch size, *num_batches* is number of batches for each teacher, *n* is the number of teachers, *num_classes* is the number of classes of each teacher, *num_epochs* is number of epochs to train, *teacher_arch* should be either {ResNet18,ResNet34,ResNet50} or any other architecture supplied, *checkpoint_dir* is a path from which teachers checkpoints will be loaded, the wanted format in this directory is (t1.pt, t2.pt,...,tn.pt), *data_dir* is a path in which the deep inverted data is stored, the wanted form should be: subdirectories of t1,t2…In each ti folder should be all the seep inverted data as batchi.pt (where i is the batch number), *save_path* is the path to save the student model in.

## Citation:
```bibtex
@inproceedings{yin2020dreaming,
	title = {Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion},
	author = {Yin, Hongxu and Molchanov, Pavlo and Alvarez, Jose M. and Li, Zhizhong and Mallya, Arun and Hoiem, Derek and Jha, Niraj K and Kautz, Jan},
	booktitle = {The IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)},
	month = June,
	year = {2020}
}
```
## Final words
Hope this repo is useful for your research. For any questions, please email guy.shapira@campus.technion.ac.il or galsidi@campus.technion.ac.il, and we will get back to you as soon as possible.
