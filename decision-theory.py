import numpy as np
import matplotlib.pyplot as plt

losses = [[15, 0],[5, 1], [1, 40],[0, 150]]
actions_names = ['Important','Show', 'Folder', 'Delete']
num_actions = len(losses)
prob_range = np.linspace(0, 1, num=600)

def expected_loss_of_action(prob_spam, action):
    losses_given_action = losses[action] # loss lunction of a particular action
    expected_loss = (1 - prob_spam)*losses_given_action[1]+prob_spam*losses_given_action[0]
    return expected_loss

# Plot the expected wasted user time for each of the four possible actions, as a function of the probability of spam:  p(spam|email) .

for action in range(num_actions):
    plt.plot(prob_range, expected_loss_of_action(prob_range, action), label=actions_names[action])

plt.xlabel('$p(spam|email)$')
plt.ylabel('Expected loss of action')
plt.legend()
plt.title('The expected wasted user time for each of the four possible actions')
plt.show()

# Plot the expected loss of the optimal action as a function of the probability of spam.

def optimal_action(prob_spam):
    expected_losses = [expected_loss_of_action(prob_spam, a) for a in range(num_actions)] #expect loss for action
    return actions_names[np.argmin(expected_losses)] # return action that yields the least expected loss

prob_range = np.linspace(0., 1., num=600)
optimal_losses = []
optimal_actions = []
for p in prob_range:
    optimal_action_p = optimal_action(p)
    optimal_action_p_ind = [i for i, a in enumerate(actions_names) if a == optimal_action_p][0]
    optimal_loss_p = expected_loss_of_action(p, optimal_action_p_ind) # expected loss of the optimal action
    optimal_actions.append(optimal_action_p) # attach optimal action and optimal loss to the list
    optimal_losses.append(optimal_loss_p)

optimal_actions = np.asarray(optimal_actions) # convert list into numpy array

important_indices = (np.argwhere(optimal_actions == "Important"))
important_start = np.min(important_indices)
important_end = np.max(important_indices)

show_indices = (np.argwhere(optimal_actions == "Show"))
show_start = np.min(show_indices)
show_end = np.max(show_indices)

folder_indices = (np.argwhere(optimal_actions == "Folder"))
folder_start = np.min(folder_indices)
folder_end = np.max(folder_indices)

delete_indices = (np.argwhere(optimal_actions == "Delete"))
delete_start = np.min(delete_indices)
delete_end = np.max(delete_indices)

plt.xlabel('p(spam|email)')
plt.ylabel('Expected loss of optimal action')
plt.plot(prob_range, optimal_losses)

plt.plot(prob_range[important_start:(important_end+1)], optimal_losses[important_start:(important_end+1)], color="b", label = 'Important')
plt.plot(prob_range[show_start:(show_end+1)], optimal_losses[show_start:(show_end+1)], color = "r", label = 'Show')
plt.plot(prob_range[folder_start:(folder_end+1)], optimal_losses[folder_start:(folder_end+1)], color = "m", label = 'Folder')
plt.plot(prob_range[delete_start:(delete_end+1)], optimal_losses[delete_start:(delete_end+1)], color = "g", label = 'Delete')
plt.legend(loc = 'upper left')
plt.title('The expected loss of the optimal action as a function of the probability of spam')
plt.show()
