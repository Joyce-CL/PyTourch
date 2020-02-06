import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
from evaluation import create_evaluation

class Trainer:
    
    def __init__(self,               
                 model,                # Model to be trained.
                 crit,                 # Loss function
                 optim = None,         # Optimiser
                 train_dl = None,      # Training data set
                 val_test_dl = None,   # Validation (or test) data set
                 cuda = True,          # Whether to use the GPU
                 early_stopping_cb = None): # The stopping criterion. 
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_cb = early_stopping_cb
        self.mode = None
        
        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients
        self._optim.zero_grad()
        # -propagate through the network
        output = self._model(x)
        # -calculate the loss
        loss = self._crit(output, y)  # y is target
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return loss

    def val_test_step(self, x, y):
        output = self._model(x)
        # propagate through the network and calculate the loss and predictions
        loss = self._crit(output, y)
        # predict
        zero_tensor = t.tensor([0, 0])
        if self._cuda:
            zero_tensor = t.tensor([0,0]).cuda()
        # if output > 0, predict_output will be 1, else will be 0
        predict_output = t.ge(output, zero_tensor).float()
        # return the loss and the predictions
        return loss, predict_output
        
    def train_epoch(self):
        # set training mode
        self.mode = 'train'
        epoch_loss = 0
        # iterate through the training set
        for img, label in self._train_dl:
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            # perform a training step
            if self._cuda:
                img = img.cuda()
                label = label.cuda()
            loss = self.train_step(img, label) # loss is a tensor with size (1, 1)
            epoch_loss += loss.item()
        # calculate the average loss for the epoch and return it
        average_loss = epoch_loss / len(self._train_dl)
        return average_loss
    
    def val_test(self):
        # set eval mode
        self.mode = 'eval'
        label_list = []
        pred_label_list = []
        epoch_loss = 0
        # disable gradient computation
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        with t.no_grad():
            for img, label in self._train_dl:
                if self._cuda:
                    img = img.cuda()
                    label = label.cuda()
                loss, predict_output = self.val_test_step(img, label)
                # save the predictions and the labels for each batch
                label_list.append(label)
                pred_label_list.append(predict_output)
                # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
                epoch_loss += loss.item()
            average_loss = epoch_loss / len(self._train_dl)
            # return the loss and print the calculated metrics
            print("\nvalidation loss: ", average_loss)
            create_evaluation(label_list, pred_label_list)
        return average_loss

    def fit(self, epochs=-1):
        assert self._early_stopping_cb is not None or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        t_list = []
        v_list = []
        epoch_counter = 0
        while True:
            # stop by epoch number
            epoch_counter += 1
            # train for a epoch and then calculate the loss and metrics on the validation set
            t_loss = self.train_epoch()
            v_loss = self.val_test()
            # append the losses to the respective lists
            t_list.append(t_loss)
            v_list.append(v_loss)
            # use the save_checkpoint function to save the model for each epoch
            self.save_checkpoint(epoch_counter)
            # check whether early stopping should be performed using the early stopping callback and stop if so
            # return the loss lists for both training and validation
            self._early_stopping_cb.step(v_loss)
            if self._early_stopping_cb.should_stop():
                return t_list, v_list



                    
        
        
        