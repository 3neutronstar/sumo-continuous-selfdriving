import collections
from rllibsumoutils.sumoutils import SUMOUtils
import logging
from pprint import pformat
logger = logging.getLogger(__name__)

class SUMOSimulationWrapper(SUMOUtils):
    """ A wrapper for the interaction with the SUMO simulation """

    def _initialize_simulation(self):
        """ Specific simulation initialization. """
        try:
            super()._initialize_simulation()
        except NotImplementedError:
            pass

    def _initialize_metrics(self):
        """ Specific metrics initialization """
        try:
            super()._initialize_metrics()
        except NotImplementedError:
            pass
        self.veh_subscriptions = dict()
        self.collisions = collections.defaultdict(int)

    def _default_step_action(self, agents):
        """ Specific code to be executed in every simulation step """
        try:
            super()._default_step_action(agents)
        except NotImplementedError:
            pass
        # get collisions
        collisions = self.traci_handler.simulation.getCollidingVehiclesIDList()
        logger.debug('Collisions: %s', pformat(collisions))
        for veh in collisions:
            self.collisions[veh] += 1
        # get subscriptions
        self.veh_subscriptions = self.traci_handler.vehicle.getAllSubscriptionResults()
        for veh, vals in self.veh_subscriptions.items():
            logger.debug('Subs: %s, %s', pformat(veh), pformat(vals))
        running = set()
        for agent in agents:
            if agent in self.veh_subscriptions:
                running.add(agent)
        if len(running) == 0:
            logger.info('All the agent left the simulation..')
            self.end_simulation()
        return True