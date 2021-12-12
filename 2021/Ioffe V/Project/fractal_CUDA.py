import pygame
import pygame as pg
import numpy as np
import math
import numba
from numba import cuda

#размер изображения <= 1000
res = W, H = 800, 500

#res = W, H = 1000, 700

#количество иттераций рендера
max_iter = 30

#красивая текстура
texture = pg.image.load('texture.jpg')
texture_size = min(texture.get_size()) - 1
texture_array = pg.surfarray.array3d(texture)

#сдвиг при нажатии клавиши
offset_shift = 2
#сдвиг изображения от центра
zoom = 2.2 / H

#количество потоков и блоков
threadsperblock = W
blockspergrid = H

class Fractal:
    def __init__(self, app):
        self.app = app
        self.screen_array = np.full((W, H, 3), [0, 0, 0], dtype=np.uint8)
        self.x = np.linspace(0, W, num=W, dtype=np.float32)
        self.y = np.linspace(0, H, num=H, dtype=np.float32)

    #---------ЗАПУСКАЕТСЯ НА ВИДЕОКАРТЕ---------
    @staticmethod
    @cuda.jit
    def render(cuda_array, texture_cuda_array, offset):
        tx = cuda.threadIdx.x
        ty = cuda.blockIdx.x
        bw = cuda.blockDim.x
        c = (tx - offset[0]) *zoom + 1j * (ty - offset[1])*zoom
        z = 0
        num_iter = 0

        for i in range(max_iter):
            z = z ** 2 + c
            if z.real ** 2 + z.imag ** 2 > 4:
                break

            num_iter += 1

        col = int(texture_size * num_iter / max_iter)
        cuda_array[tx,ty,0] = texture_cuda_array[col,col,0]
        cuda_array[tx, ty, 1] = texture_cuda_array[col, col, 1]
        cuda_array[tx, ty, 2] = texture_cuda_array[col, col, 2]



    def update(self,offset):
        cuda_array = cuda.to_device(self.screen_array)
        texture_cuda_array = cuda.to_device(texture_array)
        self.render[blockspergrid,threadsperblock](cuda_array, texture_cuda_array, offset)

        gpu_result = cuda_array.copy_to_host()
        cuda.synchronize()

        self.screen_array = gpu_result

    def draw(self):
        pg.surfarray.blit_array(self.app.screen, self.screen_array)

    def run(self,offset):
        self.update(offset)
        self.draw()


class App:
    def __init__(self):
        self.screen = pg.display.set_mode(res, pg.SCALED)
        self.clock = pg.time.Clock()
        self.fractal = Fractal(self)
        self.right_pressed = False
        self.left_pressed = False
        self.up_pressed = False
        self.down_pressed = False

        self.offset = np.array([1.3 * W, H]) // 2

    def run(self):
        while True:
            self.screen.fill('Black')
            self.fractal.run(self.offset)
            pg.display.flip()

            # обраьотка событий
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    exit()
                #ажатия клавиш
                elif event.type == pg.KEYDOWN:
                    #wasd
                    if event.key == pg.K_d:
                        self.right_pressed = True
                    if event.key == pg.K_a:
                        self.left_pressed = True
                    if event.key == pg.K_w:
                        self.up_pressed = True
                    if event.key == pg.K_s:
                        self.down_pressed = True

                    #нажатие на r сбрасывает позицию
                    if event.key == pg.K_r:
                        self.offset = np.array([1.3 * W, H]) // 2

                if event.type == pg.KEYUP:
                    if event.key == pg.K_d:
                        self.right_pressed = False
                    if event.key == pg.K_a:
                        self.left_pressed = False
                    if event.key == pg.K_w:
                        self.up_pressed = False
                    if event.key == pg.K_s:
                        self.down_pressed = False





            #print(self.offset)
            if (self.right_pressed):
                self.offset[0] -= offset_shift
            if (self.left_pressed):
                self.offset[0] += offset_shift
            if (self.up_pressed):
                self.offset[1] += offset_shift
            if (self.down_pressed):
                self.offset[1] -= offset_shift



            self.clock.tick()
            pg.display.set_caption(f'FPS: {self.clock.get_fps()}')


if __name__ == '__main__':
    app = App()
    app.run()