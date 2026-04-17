#helper classes
class DatasetInfo:
    def __init__(self, name, uci_id, columns_to_drop):
        self.name = name                      
        self.uci_id = uci_id                  
        self.columns_to_drop = columns_to_drop

class PreparedData:
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, category_names, num_features, num_classes):
        self.X_train = X_train       
        self.X_val = X_val           
        self.X_test = X_test  
        self.y_train = y_train      
        self.y_val = y_val           
        self.y_test = y_test         
        self.category_names = category_names 
        self.num_features = num_features     
        self.num_classes = num_classes       