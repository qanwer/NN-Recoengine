
The task is to predict which products a user will reorder in their next order. The evaluation metric is the F1-score between the set of predicted products and the set of true products.


## The Approach
The task was reformulated as a binary prediction task: Given a user, a product, and the user's prior purchase history, predict whether or not the given product will be reordered in the user's next order.  In short, the approach was to fit a variety of generative models to the prior data and use the internal representations from these models as features to second-level models.


### First-level models
The first-level models vary in their inputs, architectures, and objectives, resulting in a diverse set of representations.
  - **Product RNN/CNN** ([code](./models/rnn_product/rnn_product.py)): a combined RNN and CNN trained to predict the probability that a user will order a product at each timestep.  The RNN is a single-layer LSTM and the CNN is a 6-layer causal CNN with dilated convolutions.
  - **Aisle RNN** ([code](./models/rnn_aisle/rnn_aisle.py)): an RNN similar to the first model, but trained at the aisle level (predict whether a user purchases any products from a given aisle at each timestep).
  - **Department RNN** ([code](./models/rnn_department/rnn_department.py)): an RNN trained at the department level.
  - **Product RNN mixture model** ([code](./models/rnn_product/rnn_product_bmm.py)): an RNN similar to the first model, but instead trained to maximize the likelihood of a bernoulli mixture model.
  - **Order size RNN** ([code](./models/rnn_order/rnn_order_size.py)): an RNN trained to predict the next order size, minimizing RMSE.
  - **Order size RNN mixture model** ([code](./models/rnn_order/rnn_order_size_gmm.py)): an RNN trained to predict the next order size, maximizing the likelihood of a gaussian mixture model.
  - **Skip-Gram with Negative Sampling (SGNS)** ([code](./models/sgns/sgns.py)): SGNS trained on sequences of ordered products.
  - **Non-Negative Matrix Factorization (NNMF)** ([code](./models/nnmf/nnmf.py)): NNMF trained on a matrix of user-product order counts.


### Second-level models
The second-level models use the internal representations from the first-level models as features.
  - **GBM** ([https://github.com/qanwer/NN-Recoengine/blob/master/gbm_blend.py](models/blend/gbm_blend.py)): a lightgbm model.
  - **Feedforward NN** ([https://github.com/qanwer/NN-Recoengine/blob/master/nn_blend.py](models/blend/nn_blend.py)): a feedforward neural network.

The final reorder probabilities are a weighted average of the outputs from the second-level models.  The final basket is chosen by using these probabilities and choosing the product subset with maximum expected F1-score.


## Requirements
64 GB RAM and 12 GB GPU (recommended), Python 2.7

Python packages:
  - lightgbm==2.0.4
  - numpy==1.13.1
  - pandas==0.19.2
  - scikit-learn==0.18.1
  - tensorflow==1.3.0
