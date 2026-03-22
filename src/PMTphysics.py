from .physics import Stepping

#Pour l'instant, on s'intéresse aux résultats moyens (donc on travaille sur les distributions)
#Pour la géométrie et le suivi de chaque particule, à voir après

class Photocathode : 

    def __init__(self, quantum_efficiency):
        #la distribution de photon et l'efficacité quantique sont des fonctions de la longueur d'onde
        
        self.quantum_efficiency = quantum_efficiency

    def emit_electrons(self, dist_photons):
        #l est la longueur d'onde (lambda est déjà un nom dans python)
        #renvoie la distribution d'électrons créés
        def dist_elec(l):
            return dist_photons(l) * self.quantum_efficiency(l)
        return dist_elec
    

#Mettre aussi la Collection efficiency, qui peut être sûrement modélisée avec la géométrie du pb

class Dynodes :
    def __init__(self, n_dynode, gain_dynode):
        self.n_dynode = n_dynode
        self.gain_dynode = gain_dynode #fonction de lambda/énergie

    def amplify(self, dist_elec):
        #renvoie la distribution d'électrons 

        def dist_amp_elec(l):
            return dist_elec(l)* (self.gain_dynode(l)**self.n_dynode)
        
        return dist_amp_elec

#Travel time à prendre en compte

class Anode : 
    def __init__(self):
        pass

    def signal(self, dist_amp_elec): 
        return 
    


