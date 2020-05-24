from torch.nn import functional as F
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import denormalize

class GradCAM:
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers 
    target_layers = list of convolution layer index as shown in summary
    """
    def __init__(self, model, target_layers=None, num_classes=10):
        def wrap_forward_hook(layer_name):
          def forward_hook(module, input, output):
              self.activations_map[layer_name] = output.detach()
          return forward_hook

        def wrap_backward_hook(layer_name):
          def backward_hook(module, grad_in, grad_out):
              self.gradients_map[layer_name] = grad_out[0].detach()
          return backward_hook

        self.device = next(model.parameters()).device
        self.num_classes=num_classes
        self.model = model
        self.handlers = []  # a set of hook function handlers
        self.activations_map = {}
        self.gradients_map = {}
        self.target_layers = target_layers  # list

        for name, module in self.model.named_modules():
            if self.target_layers is None or name in self.target_layers:
                self.handlers.append(module.register_forward_hook(wrap_forward_hook(name)))
                self.handlers.append(module.register_backward_hook(wrap_backward_hook(name)))

    def encode_one_hot(self, target_labels=None):
        one_hots = torch.zeros_like(self.output)
        print(one_hots.shape)
        if target_labels==None:
            one_hots = torch.zeros((self.batch_size, self.num_classes)).to(self.device)
            for i in range(len(self.pred)):
                one_hots[i][self.pred[i][0]] = 1.0
        else:
            ids = target_labels.view(self.batch_size,-1).to(self.device)
            one_hots.scatter_(1,ids,1.0)
        return one_hots

    def forward(self, data):
        self.output = self.model(data)
        #self.probs = F.softmax(self.logits, dim=1)
        self.pred = self.output.argmax(dim=1, keepdim=True)

    def backward(self, target_labels):
        """
        Class-specific backpropagation
        """
        one_hot = self.encode_one_hot(target_labels)
        self.model.zero_grad()
        self.output.backward(gradient=one_hot, retain_graph=True)

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

    def find(self, maps, target_layer):
        if target_layer in maps.keys():
            return maps[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        actimaps = self.find(self.activations_map, target_layer)
        gradmaps = self.find(self.gradients_map, target_layer)
        weights = F.adaptive_avg_pool2d(gradmaps, 1)

        saliency_map = torch.mul(actimaps, weights).sum(dim=1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        # need to capture image size duign forward pass
        saliency_map = F.interpolate(
            saliency_map, (self.img_h, self.img_w), mode="bilinear", align_corners=False)

        # scale output between 0,1
        B, C, H, W = saliency_map.shape
        saliency_map = saliency_map.view(self.batch_size, -1)
        saliency_map -= saliency_map.min(dim=1, keepdim=True)[0]
        saliency_map /= saliency_map.max(dim=1, keepdim=True)[0]
        saliency_map = saliency_map.view(B, C, H, W)
        return saliency_map

    def __call__(self, data, target_labels, target_layers):
      self.model.eval()
      # map input to device
      self.batch_size, self.img_ch, self.img_h, self.img_w = data.shape
      data = data.to(self.device)
      # forward pass
      self.forward(data)
      # backward pass
      self.backward(target_labels=target_labels)
      masks_map = {}
      for i in range(len(target_layers)):
        target_layer = target_layers[i]
        print("Generating Grad-CAM @{}".format(target_layer))
        # Grad-CAM
        masks_map[target_layer]= self.generate(target_layer=target_layer)
      # remove hooks when done
      self.remove_hook()
      return masks_map, self.pred

class VisualizeCam(object):

	def __init__(self, model, classes, target_layers):
		super(VisualizeCam, self).__init__()
		self.model = model
		self.classes = classes
		self.target_layers = target_layers
		self.device = next(model.parameters()).device

		self.gcam = GradCAM(model, target_layers, len(classes))
		
	def visualize_cam(self, mask, img):
	    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
	    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
	    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
	    b, g, r = heatmap.split(1)
	    heatmap = torch.cat([r, g, b])
	    
	    result = heatmap+img.cpu()
	    result = result.div(result.max()).squeeze()
	    return heatmap, result

	def plot_heatmaps_indvidual(self, img_data, truth_class, target_class, img_name):
		fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 4),
			subplot_kw={'xticks': [], 'yticks': []})
		fig.suptitle('GradCam at different conv layers for the class: %s\nActual: %s - Predicted: %s' % (pred_class, truth_class, pred_class), fontsize=13, weight='medium', y=1.05)

		for ax, data in zip(axs.flat, img_data):
			img = data["img"]
			npimg = img.cpu().numpy()
			ax.imshow(np.transpose(npimg, (1, 2, 0)))
			ax.set_title("%s" % (data["label"]))

		plt.savefig(img_name)

	def plot_heatmaps(self, img_data, img_name):
		fig, axs = plt.subplots(nrows=len(img_data), ncols=5, figsize=(10, 50),
			subplot_kw={'xticks': [], 'yticks': []})

		for i in range(len(img_data)):
			data = img_data[i]
			for j in range(len(data)):
				img = data[j]["img"]
				npimg = img.cpu().numpy()
				axs[i][j].axis('off')
				axs[i][j].set_title(data[j]["label"])
				axs[i][j].imshow(np.transpose(npimg, (1, 2, 0)))

		fig.tight_layout()
		fig.savefig(img_name)

	def __call__(self, images, truth_inds, target_layers, mean, std, target_inds=None, metric="", per_image=True, path=None):
         masks_map, pred = self.gcam(images, target_inds, target_layers)
         if per_image:
            for i in range(min(len(images),5)):
                img = images[i]
                results_data = [{
                    "img": denormalize(img, mean, std),
                    "label": "Result:"
                }]
                heatmaps_data = [{
                    "img": denormalize(img, mean, std),
                    "label": "Heatmap:"
                }]
                for layer in target_layers:
                    mask = masks_map[layer][i]
                    heatmap, result = self.visualize_cam(mask, img)
                    results_data.append({
                        "img": result,
                        "label": layer
                    })
                    heatmaps_data.append({
                        "img": heatmap,
                        "label": layer
                    })
                pred_class = self.classes[pred[i][0]]
                truth_class = self.classes[truth_inds[i]]
                fname = path+"gradcam_%s_%s_t%s_p%s.png" % (metric, i, truth_class ,pred_class)
                self.plot_heatmaps_indvidual(results_data+heatmaps_data, truth_class,  pred_class, fname)
         else:
            img_data = []
            for i in range(len(images)):
                img = images[i]
                pred_class = self.classes[pred[i][0]]
                truth_class = self.classes[truth_inds[i]]
                results_data = [{
    				"img": denormalize(img,mean,std),
    				"label": "A:%s P:%s" % (truth_class, pred_class)
    				}]
                for layer in masks_map.keys():
                    mask = masks_map[layer][i]
                    heatmap, result = self.visualize_cam(mask, denormalize(img, mean, std))
                    results_data.append({
    					"img": result,
    					"label": "%s" % (layer)
    					})
                img_data.append(results_data)
            fname = path+"gradcam_%s.png" % (metric)
            self.plot_heatmaps(img_data, fname)



