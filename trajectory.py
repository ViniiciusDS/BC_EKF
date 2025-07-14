# trajectory.py
import numpy as np

class Trajectory:
    """
    Classe que gerencia uma sequência de waypoints a serem seguidos pelo robô.
    """
    def __init__(self, waypoints):
        """
        Args:
            waypoints (list of tuple): Lista de pontos (x,y).
        """
        self.waypoints = waypoints
        self.current_index = 0

    def get_target(self):
        """
        Retorna o waypoint atual.
        """
        if self.current_index < len(self.waypoints):
            return self.waypoints[self.current_index]
        return None

    def advance_if_reached(self, x, y, threshold=0.1):
        """
        Avança para o próximo waypoint se o robô estiver dentro do threshold.
        """
        target = self.get_target()
        if target is None:
            return
        distance = np.linalg.norm(np.array([x,y]) - np.array(target))
        if distance < threshold:
            self.current_index += 1

    def is_finished(self):
        """
        Verifica se todos os waypoints foram alcançados.
        """
        return self.current_index >= len(self.waypoints)
