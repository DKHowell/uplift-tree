import numpy as np

class UpliftTreeRegressor(object):

    def __init__(
    self,
    max_depth:int=3, # max tree depth
    min_samples_leaf: int = 1000, # min number of values in leaf
    min_samples_leaf_treated: int = 300, # min number of treatment values in leaf
    min_samples_leaf_control: int = 300):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control

    def find_thresholds(self, column_values): #finds potential cutoff values for given predictor
        unique_values = np.unique(column_values)
        if len(unique_values) > 10:
            percentiles = np.percentile(column_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
        else:
            percentiles = np.percentile(unique_values, [10, 50, 90])

        return np.unique(percentiles)

    def is_valid_split(self, left_treatment_count, left_control_count,
                       right_treatment_count, right_control_count):
        '''
        Checks if a proposed split results in a violation of min_samples_leaf,
        min_samples_leaf control, or min_samples_leaf_treated

        :param left_treatment:int - number of treated samples on left side of split
        :param left_control:int - number of control samples on left side of split
        :param right_treatment:int - number of treated samples on right side of split
        :param right_control:int - number of control samples on right side of split
        :return:bool - True if it is a valid split, else False
        '''
        if left_treatment_count + left_control_count >= self.min_samples_leaf and \
                right_treatment_count + right_control_count >= self.min_samples_leaf and \
                left_treatment_count >= self.min_samples_leaf_treated and \
                right_treatment_count >= self.min_samples_leaf_treated and \
                left_control_count >= self.min_samples_leaf_control and \
                right_control_count >= self.min_samples_leaf_control:
            return True
        return False

    def find_best_split(self, X:np.ndarray, treatment:np.ndarray, y:np.ndarray):
        '''
        finds the predictor and value to split on, by iterating through
        predictors and thresholds calculated using self.find_thresholds
        :return: split_idx:int, split_val:float
        '''
        #initiate maximum DeltaDeltaP, split_index, and split_value to be updated
        max_ddp = 0
        split_idx = None
        split_val = None

        #for each feature, calculate potential split values
        for i in range(X.shape[1]):
            threshold_options = self.find_thresholds(X[:,i])
            #for each split value, check if valid and calculate DeltaDeltaP
            for threshold in threshold_options:
                left_treatment = y[(X[:,i]<=threshold) & (treatment==1)] #treatment group on left side
                left_control = y[(X[:,i]<=threshold) & (treatment==0)] #control group on left side
                right_treatment = y[(X[:,i]>threshold) & (treatment==1)] #treatment group on right side
                right_control = y[(X[:,i]>threshold) & (treatment==0)] #control group on right side

                #check that there are enough treatment, control, and overall samples for each side of proposed split
                if self.is_valid_split(left_treatment.shape[0], left_control.shape[0],
                                       right_treatment.shape[0], right_control.shape[0]):

                    M_left = np.mean(left_control) - np.mean(left_treatment) #difference between treatment and control on left side
                    M_right = np.mean(right_control) - np.mean(right_treatment) #difference between treatment and control on right side
                    ddp = np.abs(M_left-M_right) #calculate DeltaDeltaP

                    if ddp > max_ddp: #if current split better than previous best split
                        max_ddp = ddp #update best split criterion
                        split_idx = i #record which column to use for split
                        split_val = threshold #record which value to split on

        return split_idx, split_val #return best column and value to split on

    def build_tree(self,
    X: np.ndarray, # (n * k) array with features
    treatment: np.ndarray, # (n) array with treatment flag
    y: np.ndarray, # (n) array with the target
    depth: int = 0) -> None: # current branch depth
        '''
        recursive function to build decision tree
        
        The decision tree is stored as an array, with each row representing a node
        nodes follow the format [split feature, split value, left child pointer, right child pointer]
        leaf nodes follow the format [-1 (special identifier), uplift, nan, nan]
        '''
        # create leaf node
        # Predicted value of treatment effect is difference in means between treatment and control groups
        leaf = np.array([-1, np.mean(y[treatment==1])-np.mean(y[treatment==0]), np.nan, np.nan])

        # return leaf if we have reached maximum depth
        if depth >= self.max_depth:
            return leaf

        # select feature to split on and split value
        split_idx, split_val = self.find_best_split(X, treatment, y)

        # if splitting any further would violate min_samples, return leaf
        if split_val is None:
            return leaf

        left_index = X[:, split_idx] <= split_val
        right_index = X[:, split_idx] > split_val

        # recursively build new branches
        left_tree = self.build_tree(X[left_index,:], treatment[left_index], y[left_index], depth+1)
        right_tree = self.build_tree(X[right_index,:], treatment[right_index], y[right_index], depth+1)

        # if left tree is 1d, set root node
        if len(left_tree.shape)==1:
            root = np.array([split_idx,split_val,1,2])
        # if left tree is 2d, set root node
        else:
            root = np.array([split_idx,split_val,1,left_tree.shape[0]+1])
        return np.vstack((root,left_tree,right_tree))

    def fit(
    self,
    X: np.ndarray, # (n * k) array with features
    treatment: np.ndarray, # (n) array with treatment flag
    y: np.ndarray) -> None:
        '''
        wrapper to construct decision tree using self.build_tree
        '''

        self.tree = self.build_tree(X, treatment, y)

    def find_leaf(self, point, row_num=0):
        '''
        recursive function to traverse decision tree and generate prediction

        :param point:1d ndarray - the datapoint we are predicting
        :param row_num:int - a pointer to indicate node location
        :return:float a prediction of the uplift of the point's leaf node assignment
        '''
        row_num = int(row_num)
        # if we have reached a leaf node, return regression prediction
        if int(self.tree[row_num,0]) == -1:
            return self.tree[row_num,1]
        # if feature value is <= split value, move down the left branch
        elif point[int(self.tree[row_num,0])] <= self.tree[row_num,1]:
            new_row = row_num+self.tree[row_num,2] #update row pointer with left child
        #move down the right branch
        else:
            new_row = row_num+self.tree[row_num,3] #update row pointer with right child

        return self.find_leaf(point, new_row)

    def predict(self, X: np.ndarray):
        '''
        generate predictions for an input matrix

        :param X: ndarray - (n * k) array
        :return: 1d ndarray: a vector of predictions with length n
        '''
        num_dps = X.shape[0]
        preds = np.zeros(num_dps)

        # create a 1d-array of predictions by calling find_leaf for each data point
        for i in range(num_dps):
            preds[i] = self.find_leaf(X[i]) # add prediction to array

        return preds

if __name__ == '__main__':
    x = np.load('example_X.npy')
    y = np.load('example_y.npy')
    treatment = np.load('example_treatment.npy')
    true_preds = np.load('example_preds.npy')

    UTR = UpliftTreeRegressor(max_depth=3,
    min_samples_leaf = 6000,
    min_samples_leaf_treated=2500,
    min_samples_leaf_control=2500)
    UTR.fit(x,treatment,y)
    preds = UTR.predict(x)

    print(np.sum(np.isclose(true_preds,preds)))
