import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.style.use('ggplot')

def avg(r):
    return [np.mean(r[i-30:i]) for i in range(30, len(r))]


r = np.load('reward.npy')
idx = list(range(30, len(r)))

baseline_color = (120 / 255, 10 /255, 153 / 255, 0.5)

plt.figure(num = None, figsize = (10, 6), dpi =80)
plt.title('Reward Over Time - {} Episodes'.format(len(r)))

plt.plot(r, color = (255 / 255, 99 / 255, 20 / 255, 0.35), label = 'Reward')
plt.plot(idx, avg(r), color = (255 / 255, 99 / 255, 20 / 255, 1), label = 'Average reward over 30 episodes')
plt.plot(idx, [1 for _ in idx], color = (20 / 255, 83 / 255, 255 / 255, 0.5), label = 'Baseline (average reward = 1)')
for i in range(1, 5):
    plt.plot(idx, [5 * i for _ in idx], color = baseline_color)
plt.legend()
plt.savefig('reward.png')
print(avg(r))