import matplotlib.pyplot as plt


def view_graphs(stats: list, save_path='train_plots.png'):
    """
    Function to visualize train statistics
    :param stats: list of lists [train_loss, train_iou, train_acc, val_loss, val_iou, val_acc]
    :param save_path: path to save plots
    """
    f, axes = plt.subplots(3, 1)
    axes[0].plot(stats[0], 'r', label='train')
    axes[0].plot(stats[3], 'g', label='val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(stats[1], 'r', label='train')
    axes[1].plot(stats[4], 'g', label='val')
    axes[1].set_title('mIoU')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('mIoU')
    axes[1].legend()

    axes[2].plot(stats[2], 'r', label='train')
    axes[2].plot(stats[5], 'g', label='val')
    axes[2].set_title('Accuracy')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend()

    plt.tight_layout()

    # Save plots
    plt.savefig(save_path)
