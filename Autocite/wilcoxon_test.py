# Wilcoxon signed-rank test

from scipy.stats import wilcoxon

# Example data
# Suppose each entry represents the count of good results out of 5 for each model on each x value

# Dataset used in the report from blind test. 
stat_test_dataset = [   
    [4, 3],
    [2, 2],
    [0, 0],
    [1, 1],
    [4, 2],
    [2, 3],
    [3, 3],
    [1, 0],
    [2, 1],
    [0, 0],
    [4, 3],
    [0, 0],
    [3, 2],
    [1, 1],
    [0, 2],
    [1, 0],
    [0, 0],
    [5, 5],
    [0, 0],
    [3, 3],
    [4, 2],
    [3, 2],
    [1, 1],
    [5, 5],
    [5, 4],
    [2, 2],
    [2, 2],
    [0, 0],
    [3, 3],
    [2, 2]
]

model1_results = [stat_test_dataset[i][0] for i in range(30)]
model2_results = [stat_test_dataset[i][1] for i in range(30)]

# Perform the Wilcoxon signed-rank test
stat, p_value = wilcoxon(model1_results, model2_results)

print('Statistic:', stat)
print('p-value:', p_value)

# Interpret the p-value
alpha = 0.05
if p_value < alpha:
    print("There is a significant difference between the two models.")
else:
    print("There is no significant difference between the two models.")