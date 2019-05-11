# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


from diadem.agents.buffer.buffer import Buffer
from collections import deque
import random


class Random(Buffer):

	def __init__(self, buffer_size):
		self.buffer_size = buffer_size
		self.num_experiences = 0
		self.buffer = deque()

	def sample(self, batch_size):
		# Draw ramdom batch
		return random.sample(self.buffer, batch_size)

	def size(self):
		return self.buffer_size

	def add(self, state, action, reward, next_action, done, is_demo):
		new_experience = (state, action, reward, next_action, done, is_demo)
		if self.num_experiences < self.buffer_size:
			self.buffer.append(new_experience)
			self.num_experiences += 1
		else:
			self.buffer.popleft()
			self.buffer.append(new_experience)

	def count(self):
		# If buffer is full, return buffer size
		# otherwise, return experience counter
		return self.num_experiences

	def erase(self):
		self.buffer = deque()
		self.num_experiences = 0