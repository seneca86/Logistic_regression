## Logistic regression

### Introduction

Linear regression can be generalized to handle dependent variables that are not numerical. If the dependent variable is boolean, the generalized model is called __logistic regression__. If the dependent variable is an integer count, it’s called __Poisson regression__.

As an example of logistic regression, suppose a friend of yours is pregnant and you want to predict whether the baby is a boy or a girl. You could use data from the NSFG to find factors that affect the “sex ratio” (the probability of having a boy).

If you encode the dependent variable numerically, for example 0 for a girl and 1 for a boy, in theory you could apply ordinary least squares, but not in practice:

