# Final Deep Learning project - Sports Action Recognition 

## Description
This repository concludes our final project for the ”Deep Learning” course. We present a model designed to
recognize sports players and their actions on the court, incorporating various training methods and pre-trained
models. We achieved a 60% recognition accuracy and an inference time that qualifies as
real-time recognition.
We enjoyed implementing the
diverse techniques we acquired throughout the semester and creating a project that combines both our academic
learning and personal interests.
![Example Image](https://drive.google.com/file/d/1c6SAsM3TtgFgML_3lB7YnhsI80_YcRsK/view?usp=sharing.jpg)

## Table of Contents
•⁠  ⁠[Installation](#installation)
•⁠  ⁠[Usage](#usage)
•⁠  ⁠[Contributing](#contributing)
•⁠  ⁠[License](#license)

## Installation
To install our project, first clone the code to your machine:

```
git clone https://github.com/your-username/your-repository.git
```

Then, create an enviorment for the project. A quick set-up can be accessed using the enviorment.yml and conda:

```
conda env create -f environment.yml
```

## Usage
To run the model on your own video, use:

```
python inference.py --video <path_to_your_video>
```

Please notice the model only exepcts .mp4 videos. The tagged video will be saved to the output folder.
You can also run the model on one of our examples:

```
python main.py --video "videos/Knicks3pointer.mp4"
```

If you want to train the model again, you first need to download the data set from [link](https://example.com/dataset).
Then, you can run the following command:

```
python main.py --train <output_path>
```


## Contributing and communiction
We'll be happy to answer questions and provide further information on our academic emails!
{yarin.bekor,tal.dugma,yonatan.a}@campus.technion.ac.il

## License
This project is licensed under the GPL lisence.
