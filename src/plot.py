# coding=utf-8

import torch, matplotlib.pyplot as plt

# 平滑多条数据
def smooth2d(data0, step=4):
	data = torch.zeros(data0.size())
	error = torch.ones(data0.size())
	size = data0.size(1)
	for i in range(size):
		from1 = i - step / 2
		to1 = from1 + step
		if from1 < 0: from1 = 0
		if to1 > size: to1 = size
		data1 = data0[:, from1:to1]
		data[:,i] = data1.mean(1)
		error[:,i] = data1.std(1)
	return data, error

# 平滑1条数据
def smooth1d(data0, step=3):
	data = torch.zeros(data0.size())
	error = torch.ones(data0.size())
	size = data0.size(0)
	for i in range(size):
		from1 = i - step / 2
		to1 = from1 + step
		if from1 < 0: from1 = 0
		if to1 > size: to1 = size
		data1 = data0[from1:to1]
		data[i] = data1.mean()
		error[i] = data1.std()
	return data, error

def plot(x_data, y_data, y_err=None, legends=None, title=None, xlabel=None, ylabel=None, filename=None):
	with plt.style.context(('fivethirtyeight')):
		#plt.style.use('fivethirtyeight')
		fig, ax = plt.subplots(squeeze=True)
		for i in range(y_data.size(0)):
			ax.plot(x_data.numpy(), y_data[i].numpy(), label=legends[i] if legends else '?')
			if (not y_err is None) and (not y_err[i] is None):
				ax.fill_between(x_data.numpy(), (y_data[i]-y_err[i]).numpy(), (y_data[i]+y_err[i]).numpy(), alpha=0.3)
		if legends: ax.legend(loc=2)
		if title: ax.set_title(title)
		if xlabel: ax.set_xlabel(xlabel)
		if ylabel: ax.set_ylabel(ylabel)
		#plt.subplots_adjust(left=0.1, right=0.99, top=0.92, bottom=0.12)
		fig.tight_layout()
	if filename:
		plt.savefig(filename)
	else:
		plt.show()
		# plt.close()
		# plt.show(block=False)

if __name__ == '__main__':
	import sys
	filename = sys.argv[1] if len(sys.argv) > 1 else None
	title = 'My progress'
	x_data = torch.FloatTensor([1,2,3,4,5,6,7,8,9,10])
	legends = ['C++ scores', 'Java scores', 'Python scores']
	y_data = torch.FloatTensor([
		[2,4,1,5,9,8,3,6,8,5],
		[12,14,11,15,19,18,13,16,18,15],
		[22,24,21,25,29,28,23,26,28,25],
	])
	y_data, y_err = smooth2d(y_data, step=3)
	#plt.close()
	plot(x_data, y_data, title=title, filename=filename, legends=legends, y_err=y_err, xlabel='Days', ylabel='Grade')

