import os
import os.path
import matplotlib.pyplot as plt
import wget
import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy.io
import scipy.stats
import torch 
import random
from torch import nn
from torch.distributions.normal import Normal
from functools import partial
from tqdm import trange, tqdm_notebook
from wget import download
import matplotlib.pyplot as plt

def diag_gaussian_log_density(x, mu, std):
    m = Normal(mu, std)
    return torch.sum(m.log_prob(x), axis=-1) # axis=-1 means sum over the last dimension.

# Implementing the TrueSkill Model

def log_joint_prior(zs_array):
    return diag_gaussian_log_density(zs_array, torch.tensor([0.0]), torch.tensor([1.0]))

def logp_a_beats_b(z_a, z_b):
    return -torch.logaddexp(torch.tensor([0.0]), z_b - z_a)

def log_prior_over_2_players(z1, z2):
    m = Normal(torch.tensor([0.0]), torch.tensor([[1.0]]))
    return m.log_prob(z1) + m.log_prob(z2)

def prior_over_2_players(z1, z2):
    return torch.exp(log_prior_over_2_players(z1, z2))

def log_posterior_A_beat_B(z1, z2):
    return log_prior_over_2_players(z1, z2) + logp_a_beats_b(z1, z2)

def posterior_A_beat_B(z1, z2):
    return torch.exp(log_posterior_A_beat_B(z1, z2))

def log_posterior_A_beat_B_10_times(z1, z2):
    return log_prior_over_2_players(z1, z2) + 10.0 * logp_a_beats_b(z1, z2)

def posterior_A_beat_B_10_times(z1, z2):
    return torch.exp(log_posterior_A_beat_B_10_times(z1, z2))

def log_posterior_beat_each_other_10_times(z1, z2):
    return log_prior_over_2_players(z1, z2) \
        + 10.* logp_a_beats_b(z1, z2) \
        + 10.* logp_a_beats_b(z2, z1)

def posterior_beat_each_other_10_times(z1, z2):
    return torch.exp(log_posterior_beat_each_other_10_times(z1, z2))

def plot_isocontours(ax, func, xlimits=[-4, 4], ylimits=[-4, 4], steps=101, cmap="summer"):
    x = torch.linspace(*xlimits, steps=steps)
    y = torch.linspace(*ylimits, steps=steps)
    X, Y = torch.meshgrid(x, y)
    Z = func(X, Y)
    plt.contour(X, Y, Z, cmap=cmap)
    ax.set_yticks([])
    ax.set_xticks([])

def plot_2d_fun(f, x_axis_label="", y_axis_label="", f2=None, scatter_pts=None):
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    plot_isocontours(ax, f)
    if f2 is not None:
      plot_isocontours(ax, f2, cmap='winter')
    
    if scatter_pts is not None:
      plt.scatter(scatter_pts[:,0], scatter_pts[:, 1])
    plt.plot([4, -4], [4, -4], 'b--')   # Line of equal skill
    plt.show(block=True)
    plt.draw()

# Stochastic Variational Inference on Two Players and Toy Data
    
## Complete the evidence lower bound function and the reparameterized sampler for the approximate posterior.
    
def diag_gaussian_samples(mean, log_std, num_samples):
    mean_mat = torch.stack([mean]*num_samples)
    log_std_mat = torch.stack([log_std]*num_samples)
    epsilon = torch.normal(mean = torch.tensor(0.0), std = torch.tensor(1.0), size=(num_samples, mean.shape[0]))
    return mean_mat + epsilon*torch.exp(log_std_mat)

def diag_gaussian_logpdf(x, mean, log_std):
    return diag_gaussian_log_density(x, mean, torch.exp(log_std))

def batch_elbo(logprob, mean, log_std, num_samples):
    elbo = logprob - diag_gaussian_logpdf(diag_gaussian_samples(mean, log_std, num_samples), mean, log_std)
    return elbo

## Write a loss function called  objective  that takes variational distribution parameters, and returns an unbiased 
## estimate of the negative elbo using  num_samples_per_iter  samples, to approximate the joint posterior over skills 
## conditioned on observing player A winning 10 games.

num_players = 2
n_iters = 800
stepsize = 0.0001
num_samples_per_iter = 50

def log_posterior_A_beat_B_10_times_1_arg(z1z2):
  return log_posterior_A_beat_B_10_times(z1z2[:,0], z1z2[:,1]).flatten()

def objective(params):  # The loss function to be minimized.
  return -torch.mean(batch_elbo(log_posterior_A_beat_B_10_times_1_arg(diag_gaussian_samples(params[0], params[1], num_samples_per_iter)), params[0], params[1], num_samples_per_iter)[0])

## Initialize a set of variational parameters and optimize them to approximate the joint where we observe player A winning 10 games.

def callback(params, t):
  if t % 25 == 0:
    print("Iteration {} lower bound {}".format(t, objective(params)))

# Set up optimizer.
D = 2
init_log_std  = torch.tensor([2.0]*D, requires_grad = True) # TODO.
init_mean = torch.tensor([2.0]*D, requires_grad = True) # TODO

params = (init_mean, init_log_std)
optimizer = torch.optim.SGD(params, lr=stepsize, momentum=0.9)

def update():
    optimizer.zero_grad()
    loss = objective(params)
    loss.backward()
    optimizer.step()

print("Optimizing variational parameters...")
for t in trange(0, n_iters):
    update()
    callback(params, t)


def approx_posterior_2d(z1, z2):
    # The approximate posterior
    mean, logstd = params[0].detach(), params[1].detach()
    return torch.exp(diag_gaussian_logpdf(torch.stack([z1, z2], dim=2), mean, logstd))

plot_2d_fun(posterior_A_beat_B_10_times, "Player A Skill", "Player B Skill",
            f2=approx_posterior_2d)

print("Reporting final loss: ", objective(params).item())

## Write a loss function called  objective  that takes variational distribution parameters, and 
## returns a negative elbo estimate using simple Monte carlo with  num_samples_per_iter  samples, 
## to approximate the joint where we observe player A winning 10 games and player B winning 10 games.

n_iters = 100
stepsize = 0.0001
num_samples_per_iter = 50

def log_posterior_beat_each_other_10_times_1_arg(z1z2):
    return log_posterior_beat_each_other_10_times(z1z2[:,0], z1z2[:,1]).flatten()

def objective(params):
    return -torch.mean(batch_elbo(log_posterior_beat_each_other_10_times_1_arg(diag_gaussian_samples(params[0], params[1], num_samples_per_iter)), params[0], params[1], num_samples_per_iter)[0])

def approx_posterior_2d(z1, z2):
    # The approximate posterior
    mean, logstd = params[0].detach(), params[1].detach()
    return torch.exp(diag_gaussian_logpdf(torch.stack([z1, z2], dim=2), mean, logstd))

    
init_log_std  = torch.tensor([2.0]*D, requires_grad = True) # TODO.
init_mean = torch.tensor([2.0]*D, requires_grad = True) # TODO
params = (init_mean, init_log_std)
optimizer = torch.optim.SGD(params, lr=stepsize, momentum=0.9)

print("Optimizing variational parameters...")
for t in trange(0, n_iters):
    update()
    callback(params, t)

plot_2d_fun(posterior_beat_each_other_10_times, "Player A Skill", "Player B Skill",
            f2=approx_posterior_2d)

# Approximate inference conditioned on real data

wget.download("https://raw.githubusercontent.com/pquynhvu/ML-sta2104-2023/main/chess_players.csv")
games = pd.read_csv("chess_games.csv")[["winner_index", "loser_index"]].to_numpy()
wget.download("https://raw.githubusercontent.com/pquynhvu/ML-sta2104-2023/main/chess_players.csv")
names = pd.read_csv("chess_players.csv")[["index", "player_name"]].to_numpy()

games = torch.LongTensor(games)

## Assuming all game outcomes are i.i.d. conditioned on all players' skills, the function  log_games_likelihood  
## takes a batch of player skills  zs  and a collection of observed games  games  and gives the total log-likelihood 
## for all those observations given all the skills.

def log_games_likelihood(zs, games):
    winning_player_ixs = games[:,0]
    losing_player_ixs = games[:,1]

    winning_player_skills = zs[:, winning_player_ixs] 
    losing_player_skills = zs[:, losing_player_ixs]

    log_likelihoods = logp_a_beats_b(winning_player_skills, losing_player_skills)
    return torch.sum(log_likelihoods, dim=1)

def log_joint_probability(zs):
    return log_joint_prior(zs) + log_games_likelihood(zs, games)

## Write a new objective function

num_players = 1434
n_iters = 500
stepsize = 0.0001
num_samples_per_iter = 150

def objective(params):
    return -torch.mean(batch_elbo(log_joint_probability(diag_gaussian_samples(params[0], params[1], num_samples_per_iter)), params[0], params[1], num_samples_per_iter)[0])

## Optimize, and report the final loss.

init_mean = torch.zeros(num_players, requires_grad=True)
init_log_std  = torch.zeros(num_players, requires_grad=True)
params = (init_mean, init_log_std)
optimizer = torch.optim.SGD(params, lr=stepsize, momentum=0.9)

def update():
    optimizer.zero_grad()
    loss = objective(params)
    loss.backward()
    optimizer.step()

print("Optimizing variational parameters...")
for t in trange(0, n_iters):
    update()
    callback(params, t)

print("\n Final Loss: ", objective(params).item())

## Plot the approximate mean and variance of all players, sorted by skill.

mean_skills, logstd_skills = params[0].detach(), params[1].detach()
order = torch.argsort(mean_skills)

plt.xlabel("Player Rank")
plt.ylabel("Player Skill")
plt.errorbar(range(num_players), mean_skills[order], np.exp(logstd_skills[order]**2).detach())

## List the names of the 10 players with the highest mean skill under the variational model.

names[order][-10:]

## Plot samples from the joint posterior over the skills of lelik3310 and thebestofthebad. 

lelik3310_ix = 496
thebestofthebad_ix = 512
print(names[lelik3310_ix])
print(names[thebestofthebad_ix])

fig = plt.figure(figsize=(8,8), facecolor='white')

## Label each with "<player> Skill"

plt.xlabel("lelik3310 Skill") 
plt.ylabel("thebestofthebad Skill") 
plt.plot([3, -3], [3, -3], 'b--') # Line of equal skill
samples = diag_gaussian_samples(mean_skills, logstd_skills, 100)
samples_lelik_thebest = samples[:, [lelik3310_ix, thebestofthebad_ix]]
plt.scatter(samples_lelik_thebest[:, 0], samples_lelik_thebest[:, 1])

## Compute the probability under your approximate posterior that lelik3310 has higher skill than thebestofthebad. 

def prob_A_superior_B(N, A_ix, B_ix):
    formula_est = 1 - Normal(mean_skills[A_ix] - mean_skills[B_ix], np.exp(logstd_skills[A_ix])**2 + np.exp(logstd_skills[B_ix])**2).cdf(torch.tensor(0))
    samples_A_B = np.asarray(diag_gaussian_samples(mean_skills[[A_ix, B_ix]], logstd_skills[[A_ix, B_ix]], 100))
    mc_est = np.mean(samples_A_B[:, 0] > samples_A_B[:, 1])
    return formula_est, mc_est

formula_est, mc_est = prob_A_superior_B(10000, lelik3310_ix, thebestofthebad_ix)
print(f"Exact CDF Estimate: {formula_est}")
print(f"Simple MC Estimate: {mc_est}")

## Compute the probability that lelik3310 is better than the player with the 5th lowest mean skill. 

fifth_worst_ix = order[5].item()
formula_est, mc_est = prob_A_superior_B(10000, lelik3310_ix, fifth_worst_ix)
print(f"Exact CDF Estimate: {formula_est}")
print(f"Simple MC Estimate: {mc_est}")

