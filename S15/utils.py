import json
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision

def get_classified_images(model, device, loader, classification='Miss'):
    """
    Input Args: model, device, loader, classification
        classification: Contains two parameters:- Miss, Corr
        
    Output Args:
    ImageSet - Classification Images, PredLabels - Model Predicted Labels, Target Labels
    """
    model.eval()
    ImageSet, PredLabels, TargetLabels  = torch.Tensor().to(device), torch.LongTensor().to(device), torch.LongTensor().to(device)
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).squeeze()
            correct = target.eq(pred.view_as(target))
            if classification == 'Miss':
                correct = ~correct
            ImageSet, PredLabels, TargetLabels = torch.cat((data[correct],ImageSet)), torch.cat((pred[correct], PredLabels)), torch.cat((target[correct], TargetLabels))
    return ImageSet, PredLabels, TargetLabels              

def model_builder(model_class=None, weights_path=None, local_device=torch.device("cpu")):
    if (model_class == None):
        print("Please provide the model object to be used")
        return
    local_model = model_class()
    try:
        if (weights_path != None):
            checkpoint = torch.load(weights_path)
            local_model.load_state_dict(checkpoint['model_state_dict'])           
    except:
        print("Some execption occured during loading the model")
    return local_model.to(local_device)

def class_acurracy(model, device, loader, classes):
    nb_classes=len(classes)
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    model.eval()
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = model(images.to(model.device))
            pred = outputs.argmax(dim=1, keepdim=True).squeeze()
            for t, p in zip(labels.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1    
    Accuracy = confusion_matrix.diag()/confusion_matrix.sum(1)
    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * Accuracy[i].item()))


def plot_graph(load_path, cols, legend_arr, xlabel, ylabel, title, save_path, log=None ):
    """
    Args:
        load_path : to read the file in json
        cols : Attributes to choosen from file List
        legend_arr: List containing the legends of graph
        xlabel: x name of the plot
        ylabel : y names of the plot
        title : Title of the graph
        save_path : To save the plot

    Return:
        plot the graph and save the graph by name title.
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    with open(load_path) as f:
        data = json.load(f)
    for col in cols:
        ax.plot(range(len(data[col])),data[col])
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if log:
        ax.set_yscale("log")
    ax.legend(legend_arr)
    plt.title(title)
    plt.show()
    #fig.suptitle(title, fontsize=16)
    fig.savefig(save_path+'/'+title+'.jpg')
  
 
def sample_images(imageloader, mean, std, classes, count=32, LabelClarity=False):
    dataiter = iter(imageloader)
    images, labels = dataiter.next()
    title='Sample Image Set'
    save_path = None
    # show images
    if LabelClarity ==True:
        show_classified(images, None, labels, classes, mean, std, title, save_path, count=25)
        return
    print(images.shape, torch.mean(images,[0,2,3]),torch.std(images,[0,2,3]))
    imges = denormalize(torchvision.utils.make_grid(images[:count], nrow=8), mean, std)
    npimg = imges.cpu().numpy()
    fig = plt.figure(figsize=(10,10))
    plt.figsize = (10,20)
    plt.imshow(np.transpose(npimg, (1, 2, 0)),interpolation='none')
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(count)))
  
def samples(images, mean=None, std=None, count=25):
#    if mean and std:
#        imges = torchvision.utils.make_grid(denormalize(images, mean, std)[:count], nrow=5)
#    else:
#        imges=torchvision.utils.make_grid(images[:count], nrow=5)
    imges=torchvision.utils.make_grid(images[:count], nrow=5)
    #imges=imges.permute(1,2,0)
    return imges
    
def get_meanstd(trainset, testset):
    full_data = np.concatenate((trainset.data, testset.data),axis=0)
    mean = full_data.reshape(-1, full_data.shape[-1]).mean(0)/255
    std = full_data.reshape(-1, full_data.shape[-1]).std(0)/255
    return mean, std

def denormalize(tensor, mean, std):
	single_img = False
	if tensor.ndimension() == 3:
		single_img = True 
		tensor = tensor[None,:,:,:]

	if not tensor.ndimension() == 4:
	    raise TypeError('tensor should be 4D')
	mean = torch.FloatTensor(mean).view(1, tensor.shape[1], 1, 1).expand_as(tensor).to(tensor.device)
	std = torch.FloatTensor(std).view(1, tensor.shape[1], 1, 1).expand_as(tensor).to(tensor.device)
	ret = tensor.mul(std).add(mean)
	return ret[0] if single_img else ret  

def show_classified(images, preds, actuals, classes, mean, std, title, save_path, count=25):
    fig = plt.figure(figsize=(14,16))
    columns = 5
    rows = 5
    img = denormalize(images, mean, std)
    for idx in np.arange(count):
        ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
        plt.imshow(np.transpose(img[idx].cpu().numpy(), (1, 2, 0)), interpolation='none')
        if preds == None:
            ax.set(xlabel="Actual="+classes[actuals[idx].item()])
        else:
            ax.set(ylabel="Pred="+classes[preds[idx].item()], xlabel="Actual="+classes[actuals[idx].item()])
    fig.suptitle(title, fontsize=16)
    if save_path!=None:
        fig.savefig(save_path+'/'+title+'.jpg') 