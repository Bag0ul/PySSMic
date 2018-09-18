from pykka import ThreadingActor
import optimizer


class Producer(ThreadingActor):
    def __init__(self, producers):
        super(Producer, self).__init__()
        self.optimizer = optimizer.Optimizer()
        self.consumers = []

    # Send a message to another actor in a framework agnostic way
    def send(self, message, receiver):
        pass

    # Receive a message in a framework agnostic way
    def receive(self, message, sender):
        pass

    # Function for choosing the best schedule given old jobs and the newly received one
    def optimize(self, job):
        pass

    # FRAMEWORK SPECIFIC CODE
    def on_receive(self, message):
        sender = message['sender']
        self.receive(message, sender)