import numpy as np
import math
import cv2

from numpy.core.numeric import load, zeros_like

def gaussian(xy, ctx, kernel_size):
    # https://en.wikipedia.org/wiki/Gaussian_function
    sigmax, sigmay = kernel_size, kernel_size
    result= math.exp(-1 * ((xy[0] - ctx[0])**2/(2 * sigmax) + (xy[1] - ctx[1])**2 / (2 * sigmay)))
    return result

def apply_gaussian(ctx, size, weight):
    h, w = weight.shape
    for y in range(ctx[1] - size, ctx[1] + size):
        for x in range(ctx[0] - size, ctx[1] + size):
            if x < 0 or x > w - 1 or y < 0 or y > h -1:
                continue
            weight[y, x] += gaussian((x, y), ctx, size)

def load_labelbar():
    img = cv2.imread('./matlab_jet_labelbar.jpg')
    mid_row = img.shape[0] // 2
    img = img[mid_row, :, :]
    return img

class HeatMap:
    def __init__(self, bg_img):
        self.bg_img = bg_img
        self.weight = np.zeros(bg_img.shape[:2], np.float32)
        self.weight_bg = np.zeros_like(bg_img, np.uint8)
        self.colormap = load_labelbar()

    def update(self, pos):
        self.weight *= 0
        size = 100
        apply_gaussian(pos, size, self.weight)
        idx = (self.weight * (self.colormap.shape[0] -1)).astype(np.int32).reshape(-1)
        self.weight_bg = self.colormap[idx].reshape(self.weight_bg.shape)
        # import ipdb;ipdb.set_trace()
        result = self.weight[..., None] * self.weight_bg[..., :]+ (1 - self.weight[..., None]) * self.bg_img
        self.result_img = result.astype(np.uint8)
    
    def render(self, img):
        img[:] = self.result_img[:]

def mouse_cb(event,x,y,flags,param):
    mouse_pos = param[0]
    mouse_pos[0] = x
    mouse_pos[1] = y

def test_gaussion():
    map = load_labelbar()
    size = 200
    weight = np.zeros((400, 400), np.float)
    apply_gaussian((size//2, size //2), size//2, weight)
    idx = (weight * (map.shape[0] -1)).astype(np.int32).reshape(-1)
    img = map[idx].reshape(weight.shape[0], weight.shape[0], 3)
    img = img.astype(np.uint8)
    cv2.imshow('img', img)
    cv2.waitKey(0)

def run():
    img = np.zeros((500, 500, 3), np.uint8)
    hmap = HeatMap(img)
    cv2.namedWindow('img')
    mouse_pos = np.array([0, 0])
    cv2.setMouseCallback('img', mouse_cb, param= (mouse_pos,))
    while True:
        hmap.update(mouse_pos)
        hmap.render(img)
        cv2.imshow('img', img)
        key = cv2.waitKey(1)
        if cv2.waitKey(20) & 0xFF == 27:
            break

if __name__ == '__main__':
    run()