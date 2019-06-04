# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import tensorflow as tf
from diadem.common import extract_params
from diadem.environments import Environment
from diadem.datamanager import DataManager


class AgentContext:
    def __init__(self, environment: Environment = None, datamanager: DataManager = None, preprocessor = None,
                 optimizer: tf.train.Optimizer = None, session_config=None, agent_manager=None, summary_service=None):
        self.environment = environment
        self.preprocessor = preprocessor
        self.datamanager = datamanager
        self.agent_manager = agent_manager
        self.graph = tf.Graph()
        self._session_config = session_config
        self.summary_service = summary_service
        self.session = tf.Session(
            config=session_config, graph=self.graph)
        self.optimizer = optimizer

    def clone(self, new_agent_params):
        # each agents holds its own copy, some of these are references, but for each agent a new graph is generated 
        # parameter copy clone, no instance clone, e.g. hba may need 
        environment = self.environment.__class__(
            params=new_agent_params['environment']) if new_agent_params['environment'] else self.environment

        return AgentContext(
            environment=environment,
            preprocessor=self.preprocessor,
            summary_service=self.summary_service,
            datamanager=self.datamanager,
            agent_manager=self.agent_manager,
            session_config=self._tf_session_config(new_agent_params),
            optimizer=extract_params(
                params=new_agent_params["network"]["optimizer"], object=self.optimizer)
        )

    def reset(self):
        """
        Close current session and start a new one in a new graph
        """
        #self.session.close()
        self.graph = tf.Graph()
        self.session = tf.Session(
            config=self._session_config, graph=self.graph)

    def close(self):
        #self.session.close()
        pass

    def _tf_session_config(self, params):
        allow_gpu_growth = params['allow_gpu_growth',
                                  'Allow GPU memory consumption grow dynamically and do not allocate a fixed amount']
        config = {
            'gpu_options': {
                'allow_growth': allow_gpu_growth if allow_gpu_growth else 0
            }
        }
        if params["use_only_cpu", "Use only CPU for training"]:
            config['device_count'] = {'GPU': 0}
        return tf.ConfigProto(**config)
