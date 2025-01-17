# One Pixel Attack on FashionMNIST and CIFAR-10

This repository implements the **One Pixel Attack** on two popular image classification datasets: **FashionMNIST** and **CIFAR-10**. The code includes neural network training, one-pixel attack and analysis of the results. This work is based on the [One Pixel Attack for Fooling Deep Neural Networks](https://arxiv.org/abs/1710.08864) article.


## Folders

- **OnePixelAttackFashionMNIST**: Contains the code for performing the One Pixel Attack on the FashionMNIST dataset. Inside this folder, you'll find:
  - **Network/Models**: Contains the pre-trained models and training history for FashionMNIST.
  - **AttackFashionMNIST.ipynb**: A notebook used to train the neural network and perform the One Pixel Attack on the FashionMNIST dataset.
  - **Results**: Contains the **ResultsAnalysis.ipynb** notebook, which is used to produce the analysis based on the provided data.

- **OnePixelAttackCIFAR10**: Contains the code for performing the One Pixel Attack on the CIFAR-10 dataset. Inside this folder, you'll find:
  - **Network/Models**: Contains the pre-trained models and training history for CIFAR-10.
  - **AttackCIFAR10.ipynb**: A notebook used to train the neural network and perform the One Pixel Attack on the CIFAR-10 dataset.
  - **Results**: Contains the **ResultsAnalysis.ipynb** notebook, which is used to produce the analysis based on the provided data.



## Requirements

To run the notebooks and code, you will need the following Python packages. You can install them using the provided `requirements.txt` file:

- **TensorFlow** (2.17.1)
- **Numpy** (1.26.4)
- **Keras** (3.5.0)
- **Seaborn** (0.13.2)
- **Pandas** (2.2.2)
- **Scikit-learn** (1.6.0)
- **Matplotlib** (3.8.0)

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage
1. Clone this repository to your local machine or Google Colab.
2. Install the required dependencies as described above.
3. Run the appropriate notebook (AttackFashionMNIST.ipynb or AttackCIFAR10.ipynb) to train a model and perform the one-pixel attack.
4. Analyze the results using the ResultsAnalysis.ipynb notebook.
