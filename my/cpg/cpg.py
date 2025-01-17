class Neuron:

    def __init__(self, number):
        self.number = number
        self.val = 0.0
        self.out_neigh = []

    def add_out_neigh(self, out_neigh, excitatory):
        if isinstance(out_neigh, Neuron):
            self.out_neigh.append((out_neigh, excitatory))
        else:
            raise TypeError("Out neighbors type error")

    def forward(self):
        for (nei, excitatory) in self.out_neigh:
            if excitatory:
                nei.val += 10.0  # the message is always 10.0
            else:
                nei.val -= 10.0
                nei.val = max(0, nei.val)  # inhibitory cells don't make 0 more negative


class Net:

    def __init__(self, neurons: list[Neuron]):
        self.neurons = {neuron.number: neuron for neuron in neurons}

    def add_connection(self, n1: int, n2: int, excitatory: bool):
        self.neurons[n1].add_out_neigh(self.neurons[n2], excitatory)

    def __getitem__(self, item):
        return self.neurons[item]

    def print_status(self):
        print([n.val for n in self.neurons.values()])


if __name__ == '__main__':
    ns = [Neuron(n) for n in range(6)]
    net = Net(ns)
    net.add_connection(0, 1, True)
    net.add_connection(1, 0, True)
    net.add_connection(1, 2, True)
    net.add_connection(2, 4, False)
    net.add_connection(2, 5, False)
    net.add_connection(3, 0, False)
    net.add_connection(3, 1, False)
    net.add_connection(4, 3, True)
    net.add_connection(4, 5, True)
    net.add_connection(5, 4, True)

    for i in range(100):
        cur = i % 6
        net[cur].forward()
        net.print_status()



