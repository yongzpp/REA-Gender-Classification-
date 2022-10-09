from config import cfg
from sklearn.linear_model import LogisticRegression
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

class Optimizer(object):
    '''
    Object to perform Bayes Optimization (hyperopt)
    '''
    def __init__(self):
        '''
        Initialize search space for logistic regression
        '''
        self.max_iter = cfg.optimize.max_evals
        self.space = {'C': hp.quniform('C', 0.01, 1000, 1), 
                      'solver':hp.choice ('solver', ['lbfgs', 'saga', 'newton-cg'])}
        self.solver = ['lbfgs', 'saga', 'newton-cg']
    
    def objective(self, space):
        '''
        Objective to maximize
        :param space: search space
        :return loss (negative of accuracy since minimization)
        '''
        clf = LogisticRegression(random_state=0, max_iter=10000, \
                            C=space["C"], solver=space["solver"]).fit(self.X_train, self.y_train)
        accuracy = clf.score(self.X_test, self.y_test)
        print ("SCORE:", accuracy)
        return {'loss': -accuracy, 'status': STATUS_OK }

    def optimize(self, X_train, X_test, y_train, y_test):
        '''
        :param: the data needed to run (pandas.Series)
        :return parameters of best model
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        trials = Trials()
        best_hyperparams = fmin(fn=self.objective, space=self.space, algo=tpe.suggest, \
                            max_evals=self.max_iter, trials=trials)
        print(best_hyperparams)
        return best_hyperparams["C"], self.solver[best_hyperparams["solver"]]