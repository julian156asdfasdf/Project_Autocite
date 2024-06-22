# Make plots of training data

import matplotlib.pyplot as plt
import numpy as np
import pickle

time_file_save = '2024-06-15_12-59-14' # Write the time_file_save from the training data to be plottet.
epoch_losses = pickle.load(open(f'Training_Variables/epoch_losses_{time_file_save}.pkl', 'rb'))
running_topk_accuracy = pickle.load(open(f'Training_Variables/running_topk_accuracy_{time_file_save}.pkl', 'rb'))
running_weights = pickle.load(open(f'Training_Variables/running_weights_{time_file_save}.pkl', 'rb'))

rw_shape = (len(running_weights), len(running_weights[0]),len(running_weights[0][1]))
rtka_shape = (len(running_topk_accuracy), len(running_topk_accuracy[0]))
el_shape = (len(epoch_losses), len(epoch_losses[0]))

num_weights = 764
weight_idx = np.array([250, 119, 603, 395, 118])

epochs_total = rtka_shape[0]
iterations, losses, running_epoch_losses, epochs, topk_accuracies, epochs_weight, weights = [], [], [], [], [], [], []
for i, [iteration, loss] in enumerate(epoch_losses):
    iterations.append(iteration)
    losses.append(loss)
for i in range(1,len(epoch_losses)+1):
    running_epoch_losses.append(np.mean(np.asarray(epoch_losses)[:i,1]))
for i, [epoch, topk_accuracy] in enumerate(running_topk_accuracy):
    epochs.append(epoch)
    topk_accuracies.append(topk_accuracy)
for i, [epoch, weight_list] in enumerate(running_weights):
    epochs_weight.append(epoch)
    weight_epoch = []
    for integer in weight_idx:
        weight_epoch.append(weight_list[integer])
    weights.append(weight_epoch)


fig, ax = plt.subplots()
plt.plot(iterations, running_epoch_losses, label='Running Epoch Training Loss', color='blue')
lr_text = f'lr=0.1'
ax.text(0.15, 0.5, lr_text, transform=ax.transAxes, fontsize=14, color='red')
plt.axvline(x=11.5*len(iterations)/(epochs_total-1), color='r', linestyle='--', linewidth=2)
plt.axvline(x=22.5*len(iterations)/(epochs_total-1), color='r', linestyle='--', linewidth=2)
plt.axvline(x=27.5*len(iterations)/(epochs_total-1), color='r', linestyle='--', linewidth=2)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Running Training Loss')
plot_text = f'Number of epochs: {epochs_total}\nOptimizer: Adam\nTraining size: {18000}\nLoss function: Triplet Loss'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.45, 0.95, plot_text, transform=ax.transAxes, fontsize=14,
verticalalignment='top', bbox=props)
plt.show()

fig, ax = plt.subplots()
plt.plot(epochs, topk_accuracies, label='Running Epoch Training Loss', color='blue')
lr_text = f'lr=0.1'
ax.text(0.15, 0.5, lr_text, transform=ax.transAxes, fontsize=14, color='red')
plt.axvline(x=11.5, color='r', linestyle='--', linewidth=2)
plt.axvline(x=23.5, color='r', linestyle='--', linewidth=2)
plt.axvline(x=27.5, color='r', linestyle='--', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('top-20 accuracy')
plt.title('Top-20 Accuracy During Training')
plot_text = f'Number of epochs: {epochs_total}\nOptimizer: Adam\nTraining size: {18000}\nLoss function: Triplet Loss'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.45, 0.35, plot_text, transform=ax.transAxes, fontsize=14,
verticalalignment='top', bbox=props)
plt.show()

fig, ax = plt.subplots()
for i in range(len(weight_idx)):
    plt.plot(epochs_weight, np.asarray(weights)[:,i], label=f'Weight {i}')
lr_text = f'lr=0.1'
ax.text(0.15, 0.35, lr_text, transform=ax.transAxes, fontsize=14, color='red')
plt.axvline(x=11.5, color='r', linestyle='--', linewidth=2)
plt.axvline(x=23.5, color='r', linestyle='--', linewidth=2)
plt.axvline(x=27.5, color='r', linestyle='--', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Weight value')
plt.title('Convergence of Weights')
plt.legend()
plt.show()