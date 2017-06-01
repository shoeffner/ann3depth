from matplotlib import pyplot as plt
from matplotlib import image
from mpl_toolkits.mplot3d import Axes3D


class DataBrowser:

    def __init__(self, dataset, *, rows=2):
        self.dataset = dataset
        self.rows = rows

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

            depth_axes = self.figure.add_subplot(rows, 2, 2 * i)
            depth_axes.set_xlim([0, self.dataset[0].depth.shape[1]])
            depth_axes.set_ylim([self.dataset[0].depth.shape[0], 0])
            depth_axes.set_aspect('equal')

            depth_img = depth_axes.add_image(image.AxesImage(depth_axes,
                                                             cmap='jet'))

            self.axes.append((img_axes, img, depth_axes, depth_img))

        self.show_next()

        self.figure.show()

    def show_next(self):
        self.update_axes()

    def show_previous(self):
        self.current = (self.current - 2 * self.rows) % len(self.dataset)
        self.update_axes()

    def update_axes(self):
        first = self.current
        for ia, i, da, di in self.axes:
            ia.set_title(self.dataset[self.current].name)
            i.set_data(self.dataset[self.current].img)
            ia.set_xlim([0, self.dataset[self.current].img.shape[1]])
            ia.set_ylim([self.dataset[self.current].img.shape[0], 0])

            da.set_title(self.dataset[self.current].name)
            di.set_data(self.dataset[self.current].depth)

            da.set_xlim([0, self.dataset[self.current].depth.shape[1]])
            da.set_ylim([self.dataset[self.current].depth.shape[0], 0])

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
