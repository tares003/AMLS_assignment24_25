import matplotlib.pyplot as plt
import visualtorch

from A.cnn_model import BreastCNNModel
from B.cnn_model import BloodCNNModel
from B.resnet18 import ResNet18


def visualize_model(model_class, input_shape, model_name, filename):
    # Initialize the model
    model = model_class()
    model.eval()  # Set the model to evaluation mode

    # Generate the visualization
    img = visualtorch.lenet_view(model, input_shape=input_shape)

    # Save the visualization as a PNG file
    plt.axis("off")
    plt.tight_layout()
    plt.imshow(img)
    plt.title(f"{model_name} Architecture")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"{model_name} visualization saved as {filename}.")


# Visualize BreastCNNModel
visualize_model(BreastCNNModel, input_shape=(1, 1, 128, 128), model_name="BreastCNNModel",
                filename="BreastCNNModel.png")

# Visualize BloodCNNModel
visualize_model(BloodCNNModel, input_shape=(1, 3, 128, 128), model_name="BloodCNNModel", filename="BloodCNNModel.png")

# Visualize ResNet18
visualize_model(ResNet18, input_shape=(1, 3, 224, 224), model_name="ResNet18", filename="ResNet18.png")
