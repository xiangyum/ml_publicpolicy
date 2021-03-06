\usepackage{bbm}

https://tex.stackexchange.com/questions/171711/how-to-include-latex-package-in-r-markdown?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

# Section A: Short Answers

## Q1.

Logistic regression.

SVMs do not naturally generate a probability distribution for its prediction; instead it would give you a binary yes/no answer to the question. That's not what we're looking for, since we're looking for a probability of whether the unemployment rate will go down. Logistic regressions, on the other hand, gi ve us the log odds-ratio of the two outcomes, which we can easily convert to a probability.

## Q2. 

We should do feature-scaling (or normalization?) to the data because that will speed up the gradient descent process for logistic regression.

## Q3. 

It would be the number of times a node's nearest neighbor has a different label from itself divided by the total no. of nodes.

$$ \frac{\sum_{i = 1}^{n} X_{i}}{n} $$

where $X$ is an indicator variable for when node i has a different label from its closest neighbor.

## Q4. 

Assumptions: I use Manhattan distance and uniform weights.

k = 1 has a 0% error rate whereas k = 3 has a 80% error rate. 

## Q5. 

Decision tree classifiers are the most appropriate. Decision trees partition the space into axis-parallel rectangles. In this case, 2 partitions and 4 rectangles would let us classify the data perfectly. SVMs don't work since there isn't a margin that separates the data cleanly; the same is true for logistic regression, since there isn't a decision boundary along these 2 dimensions that lets us separate the data.

## Q6.

I suppose, first I'd want to see how the two classifier types perform along the evaluation metrics I'm interested in. I don't have any a priori reason to think why  either of the two would necessarily be better. 

In the event of a tie, however, I'd choose K-NN on the basis of interpretability. First, KNN is more intuitively similar to how medical experts do diagnosis, so it'd be easy to sell them on it. Second, a deep decision tree with ~10 features would be hard to interpret.

## Q7.

It does not give you a linear classifier.. Boosting methods use a group of classifiers that improve upon one another; even if each classifier per se were a linear model, the overall boosted model would not be so.

## Q8.

Yes? Since each successive classifier explicitly tries to improve upon the previous (miss-classified observations are more highly weighted).

## Q9.

10! = 3628800

## Q10.

No. While the accuracy is high, it may not be a useful metric because there may be a large class imbalance. The precision of the model, even for the 10% its most confident about, is only about 56%. This is barely better than 50/50 guess. And the precision will only decrease as we expand our target population.

## Q11.

False. A random forest model will almost certainly tend to be better the decision tree model, but it won't *always* be better.

## Q12.

Wheen (a) information on gender is always available, (b) when gender is expected to interact with all other features.

## Q13.

When (a) information on gender may not always be available, and (b) gender is not expected to interact with all other features.

## Q14.

### Separate model:
Pros:
* It might well be the case that gender interacts with many variables that predict re-entry. Rather than creating a whole host of interaction features, we can do it by separating the genders in the very beginning.

Cons:
* The separate model assumes a greater importance of gender, since we are doing an a priori separation of the data before we even begin the analysis. We must be able to defend why this assumption is reasonable.

### Combined model
Pros:
* Gender is considered as but one of the many features. We can increase the influence of gender by constructing interaction variables if we so wished to. 

Cons:
* 

## Q15. 

I see no reason why we couldn't do both, and pick whatever performs better.

# Section B: 

## Q1.

### (a) 

Random baseline accuracy: 0.6

### (b)

Entropy: 0.971

### (c)

Information gain after split on "home insulation"

# Section C: Solving a new problem




