import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
from mpl_toolkits.mplot3d import Axes3D


class DataBrowser:

    def __init__(self, dataset, *, rows=2, key='depth', cmap='jet'):
        self.dataset = dataset
        self.rows = rows

        self.datakey = key
        self.cmap = cmap

        self.current = 0

        self.figure = plt.figure()
        self.keycb = self.figure.canvas.mpl_connect(
                'key_press_event',
                lambda event: self.__key_press_event(event))

        self.axes = []
        for i, row in enumerate(range(rows), 1):
            img_axes = self.figure.add_subplot(rows, 2, 2 * i - 1)
            img_axes.set_xlim([0, self.dataset[0].img.shape[1]])
            img_axes.set_ylim([self.dataset[0].img.shape[0], 0])
            img_axes.set_aspect('equal')

            img = img_axes.add_image(image.AxesImage(img_axes))

            key_axes = self.figure.add_subplot(rows, 2, 2 * i)
            if (hasattr(self.dataset[0], key) and
                    getattr(self.dataset[0], key) is not None):
                key_axes.set_xlim([0, getattr(self.dataset[0], key).shape[1]])
                key_axes.set_ylim([getattr(self.dataset[0], key).shape[0], 0])
            key_axes.set_aspect('equal')

            key_img = key_axes.add_image(image.AxesImage(key_axes,
                                                         cmap=self.cmap))

            self.axes.append((img_axes, img, key_axes, key_img))

        self.show_next()

        self.figure.show()

    def show_next(self):
        self.update_axes()

    def show_previous(self):
        self.current = (self.current - 2 * self.rows) % len(self.dataset)
        self.update_axes()

    def update_axes(self):
        first = self.current
        for ia, i, ka, ki in self.axes:
            ia.set_title(self.dataset[self.current].name)
            i.set_data(self.dataset[self.current].img)
            ia.set_xlim([0, self.dataset[self.current].img.shape[1]])
            ia.set_ylim([self.dataset[self.current].img.shape[0], 0])

            if getattr(self.dataset[self.current], self.datakey) is not None:
                print(getattr(self.dataset[self.current], self.datakey))
                ka.set_title(self.dataset[self.current].name)
                ki.set_data(getattr(self.dataset[self.current], self.datakey))

                ka.set_xlim([0, getattr(self.dataset[self.current],
                                        self.datakey).shape[1]])
                ka.set_ylim([getattr(self.dataset[self.current],
                                     self.datakey).shape[0], 0])
            else:
                ki.set_data(np.array([[1]]))

            self.current = (self.current + 1) % len(self.dataset)

        self.figure.suptitle(f'Showing samples {first} to {self.current - 1}')
        self.figure.canvas.draw()

    def __key_press_event(self, event):
        events = {
            'q': lambda event: plt.close(self.figure),
            'escape': lambda event: plt.close(self.figure),
            'cmd+w': lambda event: plt.close(self.figure),
            'right': lambda event: self.show_next(),
            'down': lambda event: self.show_next(),
            'left': lambda event: self.show_previous(),
            'up': lambda event: self.show_previous()
        }
        try:
            events[event.key](event)
        except KeyError:
            print(f'Key pressed but no action available: {event.key}')


if __name__ == '__main__':
    import data

    DataBrowser(data.mixed(samples=16))
    plt.show()
