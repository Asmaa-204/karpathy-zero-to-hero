## About sets

For any string s:

`set(s) == { all unique characters of s }`


- Each character is treated as an element.

- Duplicates are removed (because sets only keep unique values).

- The order is not preserved.

Examples
```py
set("hello")     # {'h', 'e', 'l', 'o'}
set("banana")    # {'b', 'a', 'n'}
set("Python")    # {'P', 'y', 't', 'h', 'o', 'n'}
set("1112233")   # {'1', '2', '3'}
set("")          # set()  (empty set)
```

So in general:
`set(string)` gives you the set of distinct characters in that string.

## Regularization:
in the counting version, we can smooth the counts by adding fake counts to all values so that no probability is zero
in the NN version, if we set all weights to be zeroes, then all logits will be zereos, then the e^logits `e^0` ==> 1 uniform probabilities

trying to force w to be 0 === label smoothing

### regularization loss:
get `W**2.sum()` or mean, if the weights are zeroes then we will have zero loss, otherwise loss will be accumulated

```py
loss = sum(-torch.log(q)).mean() + 0.01 * (W**2.sum())
```
this term `0.01 * (W**2.sum())` tries to force weights to be zeroes to minimize the loss
- ws want to be zeroes
- probs want to be uniform
- weights also want to match up probabilities given by the data

the strength of the regularization (const) is identical to the amount of the fake count you add