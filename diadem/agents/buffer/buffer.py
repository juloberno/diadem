# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


from abc import ABC, abstractmethod



class Buffer(ABC):

	@abstractmethod
	def __init__(self):
		pass
	
	@abstractmethod
	def add(self, data, **kwargs):
		pass
	
	@abstractmethod
	def sample(self, batch_size):
		pass
	