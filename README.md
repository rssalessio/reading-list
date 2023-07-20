# reading-list

This is a collection of interesting papers that I have read so far or want to read. Note that the list is not up-to-date.


## Table of Contents
1. [General Deep Learning](#general-deep-learning)
2. [Conformal Prediction](#conformal-prediction)
3. [Differential Geometry in Deep Learning](#differential-geometry-in-deep-learning)
4. [Dimensionality Reduction](#dimensionality-reduction)
5. [Thompson Sampling](#thompson-sampling)
6. [Deep Reinforcement Learning](#deep-reinforcement-learning)
7. [Reinforcement Learning](#reinforcement-learning)
8. [Bandit algorithms](#bandit-algorithms)
9. [Optimization](#optimization)
10. [Statistics](#statistics)
11. [Probability modeling and inference](#probability-modeling-inference)
12. [Books, courses and lecture notes](#books)
13. [Blogs and tutorial](#blogs)
14. [Schools](#schools)


<a name="general-deep-learning"></a>
## 1. General deep learning
* [2021, Distilling Robust and Non-Robust Features in Adversarial Examples by Information Bottleneck](https://openreview.net/pdf?id=90M-91IZ0JC)
* [2021, Slot Machines: Discovering Winning Combinations of Random Weights in Neural Networks](https://arxiv.org/pdf/2101.06475.pdf)
* [2021, Loss landscapes and optimization in over-parameterized non-linear systems and neural networks](https://arxiv.org/pdf/2003.00307.pdf)
* [2021, Why flatness correlates with generalization for Deep NN](https://arxiv.org/pdf/2103.06219.pdf)
* [2021, The Modern Mathematics of Deep Learning](https://arxiv.org/pdf/2105.04026.pdf)
* [2021, The Principles of Deep Learning Theory](https://arxiv.org/pdf/2106.10165.pdf)
* [2020, Neural tangent kernel](https://arxiv.org/pdf/1806.07572.pdf)
* [2018, Lipschitz regularity of deep neural networks: analysis and efficient estimation](https://papers.nips.cc/paper/2018/file/d54e99a6c03704e95e6965532dec148b-Paper.pdf)
* [2015, Weight Uncertainty in Neural Networks](https://arxiv.org/pdf/1505.05424.pdf)
* [1998, LeCun, Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
  
<a name="conformal-prediction"></a>
## 2. Conformal prediction
* [2022, Conformal Prediction: a Unified Reviewof Theory and New Challenges](https://arxiv.org/pdf/2005.07972.pdf)
* [2022, Conformal Off-Policy Prediction in Contextual Bandits](https://arxiv.org/pdf/2206.04405.pdf)
* [2020, Conformal Prediction Under Covariate Shift](https://arxiv.org/pdf/1904.06019.pdf)
* [2019, Conformalized Quantile Regression](https://proceedings.neurips.cc/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf)
* [2005, Algorithmic Learning in a Random World](https://link.springer.com/book/10.1007/b106715)

<a name="differential-geometry-in-deep-learning"></a>
## 3. Differential geometry in deep learning
* [2020, Neural Ordinary Differential Equations on Manifolds](https://arxiv.org/pdf/2006.06663.pdf)
* [2019, Efficient Approximation of Deep ReLU Networks for
Functions on Low Dimensional Manifolds](https://proceedings.neurips.cc/paper/2019/file/fd95ec8df5dbeea25aa8e6c808bad583-Paper.pdf)
* [2019, Diffeomorphic Learning](https://arxiv.org/pdf/1806.01240.pdf)
* [2019, Deep ReLU network approximation of functions on a manifold](https://arxiv.org/pdf/1908.00695.pdf)
* [2019, Efficient Approximation of Deep ReLU Networks for
Functions on Low Dimensional Manifolds](https://proceedings.neurips.cc/paper/2019/file/fd95ec8df5dbeea25aa8e6c808bad583-Paper.pdf)
* [2016, Deep nets for local manifold learning](https://arxiv.org/pdf/1607.07110.pdf)

<a name="dimensionality-reduction"></a>
## 4. Dimensionality reduction
* [2020, Stochastic Neighbor Embedding
with Gaussian and Student-t Distributions: Tutorial and Survey](https://arxiv.org/pdf/2009.10301.pdf)
* [2015, Parametric nonlinear dimensionality
reduction using kernel t-SNE](https://core.ac.uk/download/pdf/20074835.pdf)
* [2009, Learning a Parametric Embedding by Preserving Local Structure](http://proceedings.mlr.press/v5/maaten09a/maaten09a.pdf)

<a name="thompson-sampling"></a>
## 5. Thompson sampling
* [2020, A Tutorial on Thompson Sampling](https://arxiv.org/pdf/1707.02038.pdf)
* [2020, Neural Thompson Sampling](https://arxiv.org/pdf/2010.00827.pdf)
* [2018, Deep Contextual Multi-armed Bandits](https://arxiv.org/pdf/1807.09809.pdf)


<a name="deep-reinforcement-learning"></a>
## 6. Deep Reinforcement Learning
* ### Famous applications
  * [2022, CICERO](https://www.science.org/doi/10.1126/science.ade9097)
* ### Algorithms
  * [List of algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)
  * [DQN Pop art](https://arxiv.org/pdf/1602.07714.pdf)
  * [2018, Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf)
  * [2017, Reinforcement Learning with Deep Energy-Based Policies](https://arxiv.org/pdf/1702.08165.pdf)
* ### DQN Convergence
  * [2020, A theoretic analysis of DQN](https://arxiv.org/pdf/1901.00137.pdf)
  * [2020, A Finite-Time Analysis of Q-Learning with Neural Network Function
Approximation](http://proceedings.mlr.press/v119/xu20c/xu20c.pdf)
  * [2019, Towards Characterizing Divergence in Deep Q-Learning](https://arxiv.org/pdf/1903.08894.pdf)
* ### Soft Q learning theory
  * [2018, Equivalence Between Policy Gradients and Soft Q-Learning](https://arxiv.org/pdf/1704.06440.pdf)   
* ### Exploration papers
  * [2020, Planning go explore via self-supervised world models](https://arxiv.org/pdf/2005.05960.pdf)
  * [2017, #Exploration: A Study of Count-Based Exploration for Deep Reinforcement](https://arxiv.org/pdf/1611.04717.pdf)
  * [2016, Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/pdf/1606.01868.pdf)
  * [2022, Exploring through Random Curiosity with General Value Functions](https://arxiv.org/pdf/2211.10282.pdf)
* ### Stability
  * [2021, Enforcing Robust Control Guarantees with neural network polciies](https://arxiv.org/pdf/2011.08105.pdf)
  * [2018, Control-Theoretic Analysis of Smoothness for Stability-Certified Reinforcement Learning](https://lavaei.ieor.berkeley.edu/RL_1_2018.pdf)
* ### Dreamer algorithm
  * [2021, Mastering Atari with discrete world-models](https://arxiv.org/pdf/2010.02193.pdf)
  * [2020, Dream to control](https://arxiv.org/pdf/1912.01603.pdf)  
  * [2020, Planning to explore via supervised world-models](https://arxiv.org/pdf/2005.05960.pdf)
  * [2019, Learning latent dynamics from pixels](https://arxiv.org/pdf/1811.04551.pdf)
* ### Goal conditioned RL
  * [2021, Adversarial Intrinsic Motivation for Reinforcement learning](https://openreview.net/pdf?id=GYr3qnFKgU)
* ### Distributional RL
  * [2021, GMAC: A Distributional Perspective on Actor-Critic Framework](https://arxiv.org/pdf/2105.11366.pdf)
  * [2020, SAMPLE-BASED DISTRIBUTIONAL POLICY GRADIENT](https://arxiv.org/pdf/2001.02652.pdf)
  * [2019, Statistics and Samples in Distributional Reinforcement Learning](https://arxiv.org/pdf/1902.08102.pdf)
  * [2018, DISTRIBUTED DISTRIBUTIONAL DETERMINISTIC POLICY GRADIENTS](https://arxiv.org/pdf/1804.08617.pdf)

<a name="reinforcement-learning"></a>
## 7. Reinforcement Learning
* ### Papers
  * [2023, An Analysis of Quantile Temporal-Difference Learning](https://arxiv.org/pdf/2301.04462.pdf)
  * [2022, Understanding Policy Gradient Algorithms: A Sensitivity-Based Approach](https://proceedings.mlr.press/v162/wu22i/wu22i.pdf)
  * [2021, Sample Complexity of Asynchronous Q-Learning: Sharper Analysis and Variance Reduction](https://arxiv.org/pdf/2006.03041.pdf)
  * [2021, Adaptive Sampling for Best Policy Identification in MDPs](http://proceedings.mlr.press/v139/marjani21a/marjani21a.pdf)
  * [2020, Provably Efficient Exploration for Reinforcement Learning Using Unsupervised Learning](https://arxiv.org/pdf/2003.06898.pdf)
  * [2019, Revisiting the Softmax Bellman Operator: New Benefits and New Perspective](https://arxiv.org/pdf/1812.00456.pdf)
  * [2019, Q-learning with UCB Exploration is Sample Efficient for Infinite-Horizon MDP](https://arxiv.org/pdf/1901.09311.pdf)
  * [2019, Provably Efficient Reinforcement Learning with Linear
Function Approximation](https://arxiv.org/pdf/1907.05388.pdf)
  * [2018, Is Q-learning Provably Efficient?](https://arxiv.org/pdf/1807.03765.pdf)
  * [Adaptive sampling for policy identification](https://arxiv.org/pdf/2009.13405.pdf)
  * [On Function Approximation in Reinforcement
Learning: Optimism in the Face of Large State
Spaces](https://proceedings.neurips.cc//paper/2020/file/9fa04f87c9138de23e92582b4ce549ec-Paper.pdf)
  * [2016, Learning the Variance of the Reward-To-Go](https://jmlr.org/papers/volume17/14-335/14-335.pdf)
  * [2009, An Analysis of Reinforcement Learning with Function Approximation](http://icml2008.cs.helsinki.fi/papers/652.pdf)
  * [2008, An analysis of model-based Interval Estimation for Markov Decision Processes](https://www.sciencedirect.com/science/article/pii/S0022000008000767)
  * [2006, PAC Model-Free Reinforcement Learning](https://cseweb.ucsd.edu/~ewiewior/06efficient.pdf)
  * [2004, Bias and Variance in Value Function Estimation](https://icml.cc/Conferences/2004/proceedings/papers/248.pdf)
  * [2001, Convergence of Optimistic and Incremental Q-Learning](https://proceedings.neurips.cc/paper/2001/file/6f2688a5fce7d48c8d19762b88c32c3b-Paper.pdf)
  * [2001, TD Algorithm for the Variance of Return and Mean-Variance Reinforcement Learning](https://www.jstage.jst.go.jp/article/tjsai/16/3/16_3_353/_pdf)
  * [2000, Convergence Results for Single-Step On-Policy Reinforcement-Learning Algorithms](https://link.springer.com/content/pdf/10.1023/A:1007678930559.pdf)
  * [1993, Convergence of Stochastic Iterative Dynamic Programming Algorithms](https://proceedings.neurips.cc/paper/1993/file/5807a685d1a9ab3b599035bc566ce2b9-Paper.pdf)
  * [1992, Reinforcement learning applied to linear quadratic regulation](https://papers.nips.cc/paper/1992/file/19bc916108fc6938f52cb96f7e087941-Paper.pdf)
  * [1982,The Variance of Discounted Markov Decision Processes](https://www.jstor.org/stable/3213832)
* ### Constrained/Safe RL
  * [2022, Safety-constrained reinforcement learning with a distributional safety critic](https://link.springer.com/article/10.1007/s10994-022-06187-8)
  * [2022,Constrained Variational Policy Optimization for Safe Reinforcement Learning](https://arxiv.org/pdf/2201.11927.pdf)
  * [2022, TRC: Trust Region Conditional Value at Risk for Safe Reinforcement Learning](https://ieeexplore.ieee.org/document/9677982)
  * [2022, Towards Safe Reinforcement Learning via Constraining Conditional Value-at-Risk](https://arxiv.org/pdf/2206.04436.pdf)
  * [2022, SAAC: Safe Reinforcement Learning as an Adversarial Game of Actor-Critics](https://arxiv.org/pdf/2204.09424.pdf)
  * [2019, Benchmarking Safe Exploration in Deep Reinforcement Learning](https://cdn.openai.com/safexp-short.pdf)
  * [2017, Constrained Policy Optimization](https://arxiv.org/pdf/1705.10528.pdf)
  * [2017, Risk-Constrained Reinforcement Learning with Percentile Risk Criteria](https://arxiv.org/pdf/1512.01629.pdf)
  * [2015, Variance-Constrained Actor-Critic Algorithms for Discounted and Average Reward MDPs](https://arxiv.org/pdf/1403.6530.pdf)
  * [2015, A Comprehensive Survey on Safe Reinforcement Learning](https://www.jmlr.org/papers/volume16/garcia15a/garcia15a.pdf)
  * [2015, Risk-Sensitive and Robust Decision-Making: a CVaR Optimization Approach](https://arxiv.org/pdf/1506.02188.pdf)

* ### Off policy evaluation
  * [2022, A Review of Off-Policy Evaluation in Reinforcement Learning](https://arxiv.org/pdf/2212.06355.pdf)
  * [2022, Conformal Off-Policy Prediction in Contextual Bandits](https://arxiv.org/pdf/2206.04405.pdf)
  * [2020, CoinDICE: Off-Policy Confidence Interval Estimation](https://arxiv.org/pdf/2010.11652.pdf)
  * [2018, Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation](https://arxiv.org/pdf/1810.12429.pdf)
  * [2015, High Confidence Policy Improvement](https://people.cs.umass.edu/~pthomas/papers/Thomas2015b.pdf)
  * [2015, High Confidence Off-Policy Evaluation](https://people.cs.umass.edu/~pthomas/papers/Thomas2015.pdf)
  * [2000, Eligibility Traces for Off-Policy Policy Evaluation](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1079&context=cs_faculty_pubs)


<a name="bandit-algorithms"></a>
## 8. Bandit algorithms
* ### Papers
  * [2023, Quantile Bandits for Best Arms Identification](https://arxiv.org/pdf/2010.11568.pdf)
  * [2020, Neural Contextual Bandits with Deep Representation
and Shallow Exploration](https://arxiv.org/pdf/2012.01780.pdf)
  * [2020, Neural Contextual Bandits with UCB-based Exploration](https://arxiv.org/pdf/1911.04462.pdf)
  * [2016, Optimal Best Arm Identification with Fixed Confidence](https://arxiv.org/pdf/1602.04589.pdf)
  * [2016, On the Complexity of Best-Arm Identification in Multi-Armed Bandit Models](https://arxiv.org/pdf/1407.4443.pdf)
  * [2016, Explore First, Exploit Next: The True Shape of Regret in Bandit Problems](https://arxiv.org/abs/1602.07182)
  * [2011, Online Least Squares Estimation with Self-Normalized Processes: An Application to Bandit Problems](https://arxiv.org/pdf/1102.2670.pdf)
  * [2002, Finite-time Analysis of the Multiarmed Bandit Problem](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf)
  * [2002, Using Confidence Bounds for Exploitation-Exploration Trade-offs](https://www.jmlr.org/papers/volume3/auer02a/auer02a.pdf)
  * [2002, THE NONSTOCHASTIC MULTIARMED BANDIT PROBLEM∗](http://rob.schapire.net/papers/AuerCeFrSc01.pdf)

<a name="optimization"></a>
## 9. Optimization
### Min-max optimization
  * [2021, A mean-field analysis of two-player zero-sum games](https://arxiv.org/pdf/2002.06277.pdf)
  * [2021, The Limits of Min-Max Optimization Algorithms: Convergence to Spurious Non-Critical Sets](http://proceedings.mlr.press/v139/hsieh21a/hsieh21a-supp.pdf)
  * [2020, On the Convergence of Single-Call Stochastic Extra-Gradient Methods](https://arxiv.org/pdf/1908.08465.pdf)
  * [2020, Non-convex Min-Max Optimization: Applications, Challenges, and Recent Theoretical Advances](https://arxiv.org/pdf/2006.08141.pdf)
  * [2020, On Gradient Descent Ascent for Nonconvex-Concave Minimax Problems](http://proceedings.mlr.press/v119/lin20a/lin20a.pdf)
  * [2020, Robust Reinforcement Learning via Adversarial training with Langevin Dynamics](https://arxiv.org/pdf/2002.06063.pdf)
  * [2018, Finding Mixed Nash Equilibria of Generative Adversarial Networks](https://arxiv.org/pdf/1811.02002.pdf)
  
<a name="statistics"></a>
## 10. Statistics
  * [2022, A short note on an inequality between KL and TV](https://arxiv.org/pdf/2202.07198.pdf)
  * [2020, A Tutorial on Quantile Estimation via Monte Carlo](https://web.njit.edu/~marvin/papers/qtut-r2.pdf)
  * [2012, CONCENTRATION INEQUALITIES FOR ORDER STATISTICS](https://arxiv.org/pdf/1207.7209.pdf)
  * [1996, IMPORTANCE SAMPLING FOR MONTE CARLO ESTIMATION OF QUANTILES](https://web.stanford.edu/~glynn/papers/1996/G96.pdf)
  * [1987, Better Bootstrap Confidence Intervals](https://www.jstor.org/stable/2289144)
  * [1982, SOME METHODS FOR TESTING THE HOMOGENEITY OF RAINFALL RECORDS](https://www.sciencedirect.com/science/article/pii/002216948290066X)

<a name="probability-modeling-inference"></a>
## 11. Probability modeling and inference
  * [2021, Normalizing Flows for Probabilistic Modeling and Inference](https://jmlr.org/papers/volume22/19-1028/19-1028.pdf)

<a name="books"></a>
## 12. Lecture notes, books and courses
  * [Regularization in RL, Google, 2021](https://rl-vs.github.io/rlvs2021/class-material/regularized_mdp/Regularization_RL_RLVS.pdf)
  * [CS 6789: Foundations of Reinforcement Learning](https://wensun.github.io/CS6789.html)
  * [RL Book Theory](https://rltheorybook.github.io/)
  * [Reinforcement Learning: an introduction](http://incompleteideas.net/book/the-book.html)
  * [Bandit algorithms](https://tor-lattimore.com/downloads/book/book.pdf)
  * [2021, Lecture Notes for Statistics 311/Electrical Engineering 377](https://web.stanford.edu/class/stats311/lecture-notes.pdf)
  * [2015, Rademacher complexities and VC Dimension](http://www.cs.cmu.edu/~hanxiaol/slides/rademacher_vc_hanxiaol.pdf)
  * [2013, An introduction to stochastic approximation](http://rcombes.supelec.free.fr/pdf/lecture_stoch_approx.pdf)
  * [2006, System identification and the limits of learning from data](https://marco-campi.unibs.it/pdf-pszip/sys-id-and-limits-learning.pdf)
  * [Deep Learning, Goodfellow et al., 2016](https://www.deeplearningbook.org)
  * [The Elements of Statistical Learning, Hastie, Tibshirani, and Friedman, 2009](https://web.stanford.edu/~hastie/ElemStatLearn)
  * [Machine Learning: a Probabilistic Perspective, Murphy, 2012](https://www.cs.ubc.ca/~murphyk/MLbook/)
  * [Probability Theory: The Logic of Science, E. T. Jaynes, 2003](https://www.amazon.com/Probability-Theory-Science-T-Jaynes/dp/0521592712)
  * [CS285 at UC Berkeley, Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/)
  * [CS234 at Stanford University, Reinforcement Learning](http://web.stanford.edu/class/cs234/index.html)
  * [15.097 at MIT, Prediction: Machine Learning and Statistics](http://ocw.mit.edu/courses/sloan-school-of-management/15-097-prediction-machine-learning-and-statistics-spring-2012/)


<a name="blogs"></a>
## 13. Blogs
  * [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html)
  * [Distill's publication on Feature Visualization](https://distill.pub/2017/feature-visualization/)
  * [Lil'Log, Blog on machine learning](https://lilianweng.github.io/lil-log/)

<a name="schools"></a>
## 14. Schools
* https://www.math.unipd.it/~vargiolu/home/link.html
* [School of mathematics](http://www.smi-math.unipr.it/perugia-2021/15/)
* [Machine learning schools](https://github.com/sshkhr/awesome-mlss)
* [Prairie summer school](https://project.inria.fr/paiss/)





Papers to add

Deep Bandits Show-Off: Simple and Efficient Exploration with Deep Networks
https://proceedings.neurips.cc/paper/2016/file/abd815286ba1007abfbb8415b83ae2cf-Paper.pdf
