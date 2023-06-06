import os
import torch
import numpy as np
import scipy.misc as smp
import scipy.ndimage
from random import randint
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_one_hot(labels, num_classes, device):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
        labels : torch.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
        num_classes : int
        Number of classes
        device: string
        Device to place the new tensor on. Should be same as input
    Returns
    -------
        target : torch.Tensor on given device
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    labels=labels.unsqueeze(1)
    #print("Labels here is",labels.shape,labels.type())
    one_hot = torch.FloatTensor(labels.size(0), num_classes, labels.size(2), labels.size(3)).zero_()
    one_hot = one_hot.to(device)
    target = one_hot.scatter_(1, labels.data, 1) 
    return target

def generate_target(y_test):
    my_test_original = y_test
    my_test = np.argmax(y_test[0,:,:,:], axis = 1)
    preds = smp.toimage(my_test)
    #plt.imshow(preds, cmap='jet')
    #plt.show()
    y_target = y_test

    target_class = 13

    dilated_image = scipy.ndimage.binary_dilation(y_target[0, target_class, :, :], iterations=6).astype(y_test.dtype)

    for i in range(256):
        for j in range(256):
            y_target[0, target_class, i, j] = dilated_image[i,j]

    for i in range(256):
        for j in range(256):
            potato = np.count_nonzero(y_target[0,:,i,j])
            if (potato > 1):
                x = np.where(y_target[0, : ,i, j] > 0)
                k = x[0]
                if k[0] == target_class:
                    y_target[0,k[1],i,j] = 0.
                else:
                    y_target[0, k[0], i, j] = 0.

    my_target = np.argmax(y_target[0,:,:,:], axis = 1)
    preds = smp.toimage(my_target)
    return y_target

def generate_target_swap(y_test):
    #my_test_original = y_test
    #my_test = np.argmax(y_test[0,:,:,:], axis = -1)
    #preds = smp.toimage(my_test)
    #plt.imshow(preds, cmap='jet')
    #plt.show()

    y_target = y_test

    y_target_arg = np.argmax(y_test, axis = 1)

    y_target_arg_no_back = np.where(y_target_arg>0)

    y_target_arg = y_target_arg[y_target_arg_no_back]

    classes  = np.unique(y_target_arg)

    #print(classes)

    if len(classes) > 3:

        first_class = 0

        second_class = 0

        third_class = 0

        while first_class == second_class == third_class:
            first_class = classes[randint(0, len(classes)-1)]
            f_ind = np.where(y_target_arg==first_class)
            #print(np.shape(f_ind))

            second_class = classes[randint(0, len(classes)-1)]
            s_ind = np.where(y_target_arg == second_class)

            third_class = classes[randint(0, len(classes) - 1)]
            t_ind = np.where(y_target_arg == third_class)

            summ = np.shape(f_ind)[1] + np.shape(s_ind)[1] + np.shape(t_ind)[1]

            if summ < 1000:
                first_class = 0

                second_class = 0

                third_class = 0

        for i in range(256):
            for j in range(256):
                temp = y_target[0,second_class, i,j]
                y_target[0,second_class, i,j] = y_target[0,first_class,i,j]
                y_target[0, first_class,i, j] = temp

        '''
        print('New target')
        my_target = np.argmax(y_target[0,:,:,:], axis = -1)
        my_test = np.argmax(y_test[0, :, :, :], axis=-1)
        print('potato')
        print(np.shape(my_target))
        print(np.shape(my_test))
        #my_test = np.reshape(my_test, (256, 256))
        together = np.concatenate((my_test, my_target), axis = 1)
        preds = smp.toimage(together)
        plt.imshow(preds, cmap='jet')
        plt.show()
        '''
    else:
        y_target = y_test
        print('Not enough classes to swap!')
    return y_target


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list