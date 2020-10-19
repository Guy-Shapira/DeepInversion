# Data free model integration

[![contributions welcome](https://img.shields.io/badge/contributions-welcome-blue?style=plastic)]()
![GitHub repo size](https://img.shields.io/github/repo-size/Guy-Shapira/DeepInversion?style=plastic)

# Environment setup:
From the root folder of the project run
```
conda env create -f environment.yml
conda activate deep-inversion
```



# Implementations:
* Implemnentation of the creation process DeepInverted images from multiple pretrained models is provided in src/run_get_images.py. An example for tranformation from images to PyTorch's tensors can be seen in src/utils/image_to_batch.py
* Our Implemnentation for creation of multiple CIFAR100 teachers (split the learning task between them) is provided in src/partial_cifar_train_teacher.py. Using src/run_create_teachers.py the creation process can be automated.
* Model integration from multiple teachers is given in src/combined_model.py, the path from which teachers checkpoints are taken should be given in checkpoint_dir.


# Citation:
```bibtex
@inproceedings{yin2020dreaming,
	title = {Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion},
	author = {Yin, Hongxu and Molchanov, Pavlo and Alvarez, Jose M. and Li, Zhizhong and Mallya, Arun and Hoiem, Derek and Jha, Niraj K and Kautz, Jan},
	booktitle = {The IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)},
	month = June,
	year = {2020}
}
```
