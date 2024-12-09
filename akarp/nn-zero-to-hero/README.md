# Neural Networks: Zero to Hero

This is compliled from videos and repos of [Andrej Karpathy](https://karpathy.ai/)

A course on neural networks that starts all the way at the basics.
The course is a series of YouTube videos where we code and train neural networks together.
The Jupyter notebooks we build in the videos are then captured here.
Every lecture also has a set of exercises included in the video description. (This may grow into something more respectable).

## Lecture 1: The spelled-out intro to neural networks and backpropagation: building micrograd

Backpropagation and training of neural networks. Assumes basic knowledge of Python and a vague recollection of calculus from high school.

Original material:

- [YouTube video lecture](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Jupyter notebook files](https://github.com/karpathy/nn-zero-to-hero/lectures/micrograd)
- [micrograd Github repo](https://github.com/karpathy/micrograd)
- [colab exercises](https://colab.research.google.com/drive/1FPTx1RXtBfc4MaTkf7viZZD4U2F9gtKN?usp=sharing)

My copies of notebooks:

- [micrograd demo](micrograd_demo.ipynb)
- [micrograd trace graph](micrograd_trace_graph.ipynb)
- [video lecture notebook 1](micrograd_lecture_first_half_roughly.ipynb)
- [video lecture notebook 2](micrograd_lecture_second_half_roughly.ipynb)
- [exercises](micrograd_exercises.ipynb)

To run micrograd unit tests, run

```bash
python -m unittest discover -s micrograd_test
```

A tiny Autograd (automatic gradient) engine (with a bite! :). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows. Potentially useful for educational purposes.

The notebook `micrograd_demo.ipynb` provides a full demo of training an 2-layer neural network (MLP) binary classifier. This is achieved by initializing a neural net from micrograd.nn module, implementing a simple svm "max-margin" binary classification loss and using SGD for optimization. As shown in the notebook, using a 2-layer neural net with two 16-node hidden layers we achieve a good decision boundary on the moon dataset.

For added convenience, the notebook `micrograd_trace_graph.ipynb` produces graphviz visualizations.

## Lecture 2: The spelled-out intro to language modeling: building makemore

We implement a bigram character-level language model, which we will further complexify in followup videos into a modern Transformer language model, like GPT. In this video, the focus is on (1) introducing torch.Tensor and its subtleties and use in efficiently evaluating neural networks and (2) the overall framework of language modeling that includes model training, sampling, and the evaluation of a loss (e.g. the negative log likelihood for classification).

Original material:

- [YouTube video lecture](https://www.youtube.com/watch?v=PaCmpygFfXo)
- [Jupyter notebook files](https://github.com/karpathy/nn-zero-to-hero/lectures/makemore/makemore_part1_bigrams.ipynb)
- [makemore Github repo](https://github.com/karpathy/makemore)

My copies of notebooks:

- [video lecture notebook 1](makemore_part1_bigrams.ipynb)

`makemore` takes one text file as input, where each line is assumed to be one training thing, and generates more things like it.
Under the hood, it is an autoregressive character-level language model, with a wide choice of models from bigrams all the way to a Transformer (exactly as seen in GPT).
For example, we can feed it a database of names, and makemore will generate cool baby name ideas that all sound name-like, but are not already existing names.
Or if we feed it a database of company names then we can generate new ideas for a name of a company.
Or we can just feed it valid scrabble words and generate english-like babble.

This is not meant to be too heavyweight library with a billion switches and knobs.
It is one hackable file, and is mostly intended for educational purposes. PyTorch is the only requirement.

Exercises:

- E01: train a trigram language model, i.e. take two characters as an input to predict the 3rd one. Feel free to use either counting or a neural net Evaluate the loss; Did it improve over a bigram model?
- E02: split up the dataset randomly into 80% train set, 10% dev set, 10% test set. Train the bigram and trigram models only on the training set. Evaluate them on dev and test splits. What can you see?
- E03: use the dev set to tune the strength of smoothing (or regularization) for the trigram model - i.e. try many possibilities and see which one works best based on the dev set loss. What patterns can you see in the train and dev set loss as you tune this strength? Take the best setting of the smoothing and evaluate on the test set once and at the end. How good of a loss do you achieve?
- E04: we saw that our 1-hot vectors merely select a row of W, so producing these vectors explicitly feels wasteful. Can you delete our use of F.one_hot in favor of simply indexing into rows of W?
- E05: look up and use F.cross_entropy instead. You should achieve the same result. Can you think of why we'd prefer to use F.cross_entropy instead?
- E06: meta-exercise! Think of a fun/interesting exercise and complete it.

## Lecture 3: Building makemore Part 2: MLP

We implement a multilayer perceptron (MLP) character-level language model.
In this video we also introduce many basics of machine learning (e.g. model training, learning rate tuning, hyperparameters, evaluation, train/dev/test splits, under/overfitting, etc.).

Original material:

- [YouTube video lecture](https://youtu.be/TCH_1BHY58I)
- [Jupyter notebook files](https://github.com/karpathy/nn-zero-to-hero/lectures/makemore/makemore_part2_mlp.ipynb)
- [Colab notebook](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbHRGbXpZS0htemZCVV9ValJ3NEhGaU9NX1dCZ3xBQ3Jtc0tuaUxBemdrNlBkS2dRMGNqRnByLTBSX1MyMF9McGpMamN2SXo1eC00ZGJkVnBOMGwwUk1hN2NNNlFaZE0zUGIyR09Ub202akJEMFRVclRIWkFnLTFlOGRCNW1nTHN2cVZtVnZnaFdMakhWYTdncmc3WQ&q=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1YIfmkftLrz6MPTOO9Vwqrop2Q5llHIGK%3Fusp%3Dsharing&v=TCH_1BHY58I)
- [makemore Github repo](https://github.com/karpathy/makemore)

My copies of notebooks:

- [video lecture notebook 2](makemore_part2_mlp.ipynb)
- [Colab notebook](build_makemore_mlp.ipynb)

Based on article [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), [local copy](bengio03a - A Neural Probabilistic Language Model.pdf)

## Lecture 4: Building makemore Part 3: Activations & Gradients, BatchNorm

We dive into some of the internals of MLPs with multiple layers and scrutinize the statistics of the forward pass activations, backward pass gradients, and some of the pitfalls when they are improperly scaled. We also look at the typical diagnostic tools and visualizations you'd want to use to understand the health of your deep network. We learn why training deep neural nets can be fragile and introduce the first modern innovation that made doing so much easier: Batch Normalization. Residual connections and the Adam optimizer remain notable todos for later video.

- [YouTube video lecture](https://youtu.be/P6sfmUTpUmc)
- [Jupyter notebook files](lectures/makemore/makemore_part3_bn.ipynb)
- [makemore Github repo](https://github.com/karpathy/makemore)

## Lecture 5: Building makemore Part 4: Becoming a Backprop Ninja

We take the 2-layer MLP (with BatchNorm) from the previous video and backpropagate through it manually without using PyTorch autograd's loss.backward(). That is, we backprop through the cross entropy loss, 2nd linear layer, tanh, batchnorm, 1st linear layer, and the embedding table. Along the way, we get an intuitive understanding about how gradients flow backwards through the compute graph and on the level of efficient Tensors, not just individual scalars like in micrograd. This helps build competence and intuition around how neural nets are optimized and sets you up to more confidently innovate on and debug modern neural networks.

I recommend you work through the exercise yourself but work with it in tandem and whenever you are stuck unpause the video and see me give away the answer. This video is not super intended to be simply watched. The exercise is [here as a Google Colab](https://colab.research.google.com/drive/1WV2oi2fh9XXyldh02wupFQX0wh5ZC-z-?usp=sharing). Good luck :)

- [YouTube video lecture](https://youtu.be/q8SA3rM6ckI)
- [Jupyter notebook files](lectures/makemore/makemore_part4_backprop.ipynb)
- [makemore Github repo](https://github.com/karpathy/makemore)

## Lecture 6: Building makemore Part 5: Building WaveNet

We take the 2-layer MLP from previous video and make it deeper with a tree-like structure, arriving at a convolutional neural network architecture similar to the WaveNet (2016) from DeepMind. In the WaveNet paper, the same hierarchical architecture is implemented more efficiently using causal dilated convolutions (not yet covered). Along the way we get a better sense of torch.nn and what it is and how it works under the hood, and what a typical deep learning development process looks like (a lot of reading of documentation, keeping track of multidimensional tensor shapes, moving between jupyter notebooks and repository code, ...).

- [YouTube video lecture](https://youtu.be/t3YJ5hKiMQ0)
- [Jupyter notebook files](lectures/makemore/makemore_part5_cnn1.ipynb)

## Lecture 7: Let's build GPT: from scratch, in code, spelled out.

We build a Generatively Pretrained Transformer (GPT), following the paper "Attention is All You Need" and OpenAI's GPT-2 / GPT-3. We talk about connections to ChatGPT, which has taken the world by storm. We watch GitHub Copilot, itself a GPT, help us write a GPT (meta :D!) . I recommend people watch the earlier makemore videos to get comfortable with the autoregressive language modeling framework and basics of tensors and PyTorch nn, which we take for granted in this video.

- [YouTube video lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY). For all other links see the video description.

## Lecture 8: Let's build the GPT Tokenizer

The Tokenizer is a necessary and pervasive component of Large Language Models (LLMs), where it translates between strings and tokens (text chunks). Tokenizers are a completely separate stage of the LLM pipeline: they have their own training sets, training algorithms (Byte Pair Encoding), and after training implement two fundamental functions: encode() from strings to tokens, and decode() back from tokens to strings. In this lecture we build from scratch the Tokenizer used in the GPT series from OpenAI. In the process, we will see that a lot of weird behaviors and problems of LLMs actually trace back to tokenization. We'll go through a number of these issues, discuss why tokenization is at fault, and why someone out there ideally finds a way to delete this stage entirely.

- [YouTube video lecture](https://www.youtube.com/watch?v=zduSFxRajkE)
- [minBPE code](https://github.com/karpathy/minbpe)
- [Google Colab](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing)

---
Ongoing...
**License** MIT
