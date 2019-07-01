from keras import layers
from keras import backend as K
import keras
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
from VAE_version3 import VariationalAutoEncoder

make_plot = False

def scatter_plot(z_mean, y_test, x_test, title='', cmap='PRGn'):
    # Plot just how the latent space reacts to mix data
    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig)
    p = ax.scatter(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2], c=y_test, cmap=plt.get_cmap(cmap))
    fig.colorbar(p, fraction=0.060)
    plt.title(title)

    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):

        pos = p.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}".format(" ".join(list(map(str, ind["ind"]))))

        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = p.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)

                if event.key == 'x':
                    val = ind['ind']
                    print(val[0])
                    plt.figure()
                    plt.title(str(val[0]))
                    plt.imshow(x_test[val[0]].reshape(28, 28))
                    plt.gray()
                    plt.show()
                    print('KEY')


                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()


    fig.canvas.mpl_connect("motion_notify_event", hover)
    fig.canvas.mpl_connect('key_press_event', hover)

    fig.show()


def plot_reconstructions(model, x_test, title='', n=10):
    # Plot comparisons between original and decoded images with test data
    decoded_imgs = model.VAE.predict(x_test)
    plt.figure(figsize=(20, 4))

    # selecxt n random images to display

    selection = np.random.randint(decoded_imgs.shape[0], size=n)

    for i in range(n):
        # disp original

        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[selection[i]].reshape(28, 28))
        plt.gray()

        ax = plt.subplot(2, n, i + n + 1)
        plt.imshow(decoded_imgs[selection[i]].reshape(28, 28))
        plt.gray()

    plt.title(title)
    plt.show()



VAE = VariationalAutoEncoder(img_shape=(28, 28, 1), latent_space_dims=3, batch_size=16)
VAE.data_prep(keep_labels=[8])
VAE.load_weights(full_path='..\mnist_AE\models/trained_digits_8')


x_test_seen, y_test_seen = [VAE.x_test, VAE.y_test]

VAE.data_prep(keep_labels=[1, 0, 5])
x_test_unseen, y_test_unseen = [VAE.x_test, VAE.y_test]

VAE.data_prep(keep_labels=[0, 1, 5, 8])
x_test_mix, y_test_mix = [VAE.x_test, VAE.y_test]


# PLOTTING & METRICS===================================================================================================

# plot the latent space of the VAE
# encoder = Model(input_img, [z_mean, z_log_var, latent_space], name='encoder')

z_mean_unseen, _, = VAE.encoder.predict(x_test_unseen, batch_size=16)
z_mean_seen, _, = VAE.encoder.predict(x_test_seen, batch_size=16)
z_mean_mix, _, = VAE.encoder.predict(x_test_mix, batch_size=16)

scatter_plot(z_mean=z_mean_mix, y_test=y_test_mix, x_test=x_test_mix,
             title='Latent Space THROUGH PREDICTION: Seen & unseen data')

scatter_plot(z_mean=z_mean_seen, y_test=y_test_seen, x_test=x_test_seen,
             title='Latent Space: Seen data')

scatter_plot(z_mean=z_mean_unseen, y_test=y_test_unseen, x_test=x_test_unseen,
             title='Latent Space: Unseen data')


# Plot just how the latent space reacts to both seen andunseen overlayed data
fig = plt.figure(figsize=(6, 6))
ax = Axes3D(fig)
p = ax.scatter(z_mean_seen[:, 0], z_mean_seen[:, 1], z_mean_seen[:, 2], c=y_test_seen, cmap=plt.get_cmap('PRGn'))
d = ax.scatter(z_mean_unseen[:, 0], z_mean_unseen[:, 1], z_mean_unseen[:, 2], c=y_test_unseen, cmap=plt.get_cmap('bwr'))
fig.colorbar(p, fraction=0.060)
fig.colorbar(d, fraction=0.060)
plt.title('Latent Space OVERLAY: Unseen & Seen data')
fig.show()


plot_reconstructions(model=VAE, x_test=x_test_mix, title='mix')
plot_reconstructions(model=VAE, x_test=x_test_unseen, title='unseen')
plot_reconstructions(model=VAE, x_test=x_test_seen, title='mix')


seen_RE = VAE.VAE.evaluate(x_test_seen, x_test_seen, batch_size=16)

unseen_RE =  VAE.VAE.evaluate(x_test_unseen, x_test_unseen, batch_size=16)


plt.figure(figsize=(6, 6))
plt.bar(x=np.arange(len([seen_RE, unseen_RE])), height=[seen_RE, unseen_RE], tick_label=['seen_RE', 'unseen_RE'])
plt.show()

