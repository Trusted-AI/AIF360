class ClassifierHistory(object):

    def append_classifier(self, classifier):
        self.classifiers.append(classifier)

    def get_most_recent(self):
        return self.classifiers[-1]

    """docstring for ClassifierHistory"""
    def __init__(self):
        super(ClassifierHistory, self).__init__()
        self.classifiers = []
        self.predictions = []
        