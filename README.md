## Fundamentals of Recommender Systems
Implementation of [Recommender Systems](http://csse.szu.edu.cn/staff/panwk/recommendation/).

Dataset: [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)

Optimizer: Stochastic Gradient Descent (SGD)

See code example: [Factorizing Personalized Markov Chains](https://github.com/weixinchen98/recommendation/blob/main/Recommendation%20with%20sequential%20feedback/Factorizing%20Personalized%20Markov%20Chains/implement.py) (using tensorflow: [Tensorflow Example](https://github.com/weixinchen98/recommendation/blob/main/Recommendation%20with%20sequential%20feedback/Factorizing%20Personalized%20Markov%20Chains/tf_implement.py))


### Recommendation with explicit feedback (Multi-class feedback)

Evaluated by Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) as the distance between the real ratings and the predicted ratings. The calculation of MAE and RMSE as below:

![equation](https://latex.codecogs.com/gif.latex?MAE&space;=&space;\sum_{(u,&space;i,&space;r_{ui})&space;\in&space;R^{te}&space;}&space;|&space;r_{ui}&space;-&space;\widehat{r}_{ui}&space;|&space;/&space;|R^{te}|)

![equation](https://latex.codecogs.com/gif.latex?RMSE&space;=&space;\sqrt{\sum_{(u,&space;i,&space;r_{ui})&space;\in&space;R^{te}&space;}&space;(&space;r_{ui}&space;-&space;\widehat{r}_{ui}&space;)&space;^&space;2&space;/&space;|R^{te}|})

Implemented methods regarding this subject:
- Average Filling
- Memory-Based Collaborative Filtering
- Matrix Factorization
- SVD++
- Matrix Factorization with Multiclass Preference Context

### Recommendation with implicit feedback (One-class feedback)

Evaluated by Ranking-Oriented Evaluation Metrics as the rationality of ranking, e.g. the precision denotes the proportion of recommended items in the test set. The full implementation of [Ranking Evaluation](https://github.com/weixinchen98/recommendation/blob/main/utils/ranking_evaluation.py).

Implemented methods regarding this subject:
- Ranking-Oriented Evaluation Metrics
- Memory-Based One-Class Collaborative Filtering
- Bayesian Personalized Ranking
- Factored Item Similarity Models with RMSE Loss
- Factored Item Similarity Models with AUC Loss
- Matrix Factorization with Logistic Loss

### Recommendation with sequential feedback
Take user-interacted items sorted by time sequence as input and evaluated by Ranking-Oriented Evaluation Metrics.

Implemented methods regarding this subject:
- Factorizing Personalized Markov Chains
- Self-Attentive Sequential Recommendation


